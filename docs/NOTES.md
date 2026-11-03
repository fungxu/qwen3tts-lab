# 工程笔记

搭建与调试过程中确认的事实、踩过的坑，以及还没解决的问题。写给未来的自己，也写给想复现的人。

- 环境：M4 Pro / 48 GB / macOS 26.5.1 / Python 3.12
- 上游：[QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)（Apache 2.0）
- 起点：报告《AI 数字人全流程》第三章「造声音」

## 一、模型事实

| 项 | 值 |
|---|---|
| 家族 | 1.7B / 0.6B × { CustomVoice, VoiceDesign, Base } |
| 主权重 | 3.83 GB（bf16）+ 0.68 GB speech_tokenizer ≈ 4.5 GB/个 |
| 输出 | 24 kHz，16-bit PCM |
| 音色（CustomVoice） | Vivian / Serena / Uncle_Fu / Dylan(京) / Eric(川) / Ryan / Aiden / Ono_Anna / Sohee |
| 语言 | 10 国语言 + 2 方言，`language` 大小写不敏感，`Auto` 自动判别 |
| 码率 | 12 Hz tokenizer，`position_id_per_seconds: 13` |

三者的区别只在 infer 入口：

```python
model.generate_custom_voice(text=..., speaker="Serena", language="Chinese", instruct=...)
model.generate_voice_design (text=..., instruct=..., language="Chinese")
model.generate_voice_clone  (text=..., ref_audio=..., ref_text=..., x_vector_only=False)
```

`x_vector_only=True` 可以省掉 `ref_text`（不问参考音说了什么，只取音色向量），代价是韵律迁移变弱。

## 二、Mac 能不能跑

能。源码里对非 CUDA 设备做了回退（attention mask 在 CPU 上构建），`device_map="mps"` 直接可用，不需要改代码。

实测（6.2 s 音频，热态）：

| 精度 | 加载 | 生成 | RTF |
|---|---|---|---|
| bfloat16 | 3.6 s | 13–15 s | **2.2–2.5×** |
| float32 | 7.8 s | 24.6 s | 3.97× |

- bf16 比 fp32 快约 1.7×，且 Whisper 回听**逐字一致**（字面相似度 1.000），所以默认 bf16
- 首次生成比热态慢 30–50%：MPS 算子图首次编译，属正常
- 模型加载 3–19 s（冷/热文件缓存差异大）

## 三、踩过的坑

### 1. `uv pip install -e` 装 qwen-tts 报 EEXIST

上游没发布 PyPI 包，只能源码引入。用 uv 做 editable 安装会撞上 uv build 缓存损坏（`EEXIST`）。

绕法：不装，直接把源码目录写进 `.pth`：

```bash
SP=$(.venv/bin/python -c "import site; print(site.getsitepackages()[0])")
echo "$PWD/_src" > "$SP/qwen_tts_src.pth"
```

`setup.sh` 里就是这么做的。

### 2. faster-whisper 新版不能直接吃文件路径

1.2.1 配新版 `av` 时，`decode_audio(path)` 的 `metadata_errors` 参数没了，传路径会抛异常。

绕法：自己解码成 numpy 再喂。

```python
x, sr = sf.read(path)
y = librosa.resample(x.astype("float32"), orig_sr=sr, target_sr=16000)
segs, _ = model.transcribe(y, language="zh")
```

### 3. `@app.on_event("startup")` 已弃用

starlette 1.10 起必须用 lifespan。预热逻辑改成 `@asynccontextmanager` + 后台线程，避免阻塞首个请求。

### 4. 沙箱里 `Path.mkdir(exist_ok=True)` 会抛 EEXIST

某些沙箱环境代理了 `mkdir`，即使带 `exist_ok=True` 也抛 `FileExistsError`。包一层 try/except 即可。

### 5. 磁盘是真正的瓶颈

三个模型合计 11.5 GB，加 venv 约 1.5 GB。第一次搭的时候磁盘只剩 2.1 GB，被 31 GB 的 swapfile 占满（`/System/Volumes/VM`，只有重启才释放）。

教训：**macOS 的 `df` 显示"可用"不等于真能写**，长时间跑推理会让 swap 疯涨；装模型前先确认 `/System/Volumes/VM` 的占用。

### 6. 下载中断会留下 `.part`

`download_model.py` 支持 Range 续传，但需要显式重跑同一条命令。中断后 `model.safetensors` 不存在、只有 `.part` 时，服务会判定该模式「不可用」——重跑下载补齐即可，校验值记在 `CHECKSUMS.json`。

### 7. pysox 的噪音警告

没装 `sox` 命令行时，`import sox` 会打印一屏 "SoX could not be found!"。只有 25 Hz tokenizer 真用到 sox，12 Hz 链路不受影响。想消掉警告：`brew install sox`。

## 四、还没解决的问题

**起音噪声。** CustomVoice 链路偶发在句首多出半个音节，Whisper 会听成「喂」「阿姨」之类。用中性描述（不带"慵懒""气声"）也会出现，剪掉首尾静音并不能消除——说明它是模型真的生成了额外的音素，不是底噪。

目前的对策：在意开头干净度时走克隆链路（未复现该问题）。

**设计音色会漂移。** VoiceDesign 每次调用重新采样，同一段描述、不同台词，音高能差 20–30 Hz。要固定就用闭环：设计 → 挑一个存成音色卡 → 用 Base 克隆。这也是报告里推荐的流程。

**没有流式。** 整句合成完才返回，长文本体验差。上游提供了流式接口，但按句切分 + 边出边播需要改前后端协议，暂未做。

## 五、复现

```bash
./setup.sh                                          # 环境
./run.sh                                            # 起服务（自动补模型）
DEV=mps DT=bfloat16 .venv/bin/python tts_bench.py   # 速度基准
python scripts/check.py                             # 静态自检
```
