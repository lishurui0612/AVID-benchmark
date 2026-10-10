# AVID benchmark

本仓库包含两个主要部分：独立评测函数，以及 LDM、MindEye1、MindEye2、MindBridge 的训练与预测代码。假设已下载发布的数据，并已准备好预训练权重；代码直接读取配置中的本地路径。

## 使用

Python 3.10。基础评分安装 `pip install -e .`；图像 encoder 安装 `pip install -e '.[evaluation]'`。四个 baseline 还需要模型依赖，见 `requirements/models.txt`。Qwen3-VL 属性抽取和 TIFA 使用各自兼容的环境；已保存的答案可以在基础环境中直接评分。

编辑 `config.yaml` 的数据、资产和输出路径。模型相关路径可以通过 `model_overrides` 设置，不需要搬动或重复下载原来的权重。代码强制 HF 离线模式，缺少所需资产时报告具体路径。

模板中 MindEye1/MindBridge 的预测使用历史 img2img 配置（`img2img_strength: 0.85`）。将 `autoencoder_checkpoint_path` 指向对应 AVID 被试、刺激类型和 vertex ROI 的已有低层权重；其输入维度须与原始数据匹配。MindBridge 的低层模型读取原始 vertex，高层模型单独池化。缺少权重时会报错。只使用 high-level pipeline 时，可明确设置 `img2img_strength: 1.0`。

```bash
python train.py --model mindeye1 --subject 1 --stimulus complex --config config.yaml
python predict.py --model mindeye1 --subject 1 --stimulus complex --checkpoint /path/to/best.pth --config config.yaml
python evaluate.py --predictions /path/to/manifest.jsonl --metrics pixcorr ssim inception_2way clip_i_2way clip_t_2way --config config.yaml --output scores.json
```

三个命令分别训练、预测和评测，不串联执行。训练期间允许计算验证损失以选择检查点，但关闭生成图像和最终 benchmark 评分。每个模型都有独立的 `avid_benchmark/recipes/<model>.json`，安装后也可以直接使用默认配方。`--recipe` 可以指定修改后的配方，`--epochs 1` 仅用于训练 smoke 检查。配置默认读取当前工作目录的 `config.yaml`；预测可用 `--max-samples 8` 做小规模检查，默认 0 使用完整测试集。

MindEye1 高层和低层可以分别训练。高层使用上面的默认命令；低层使用 `python train.py --model mindeye1 --component lowlevel --subject 1 --stimulus complex --config config.yaml`，默认配方为 `mindeye1_lowlevel.json`（120 epochs、batch16）。低层从随机初始化训练完整 vertex 输入，使用原 AVID 的 latent L1 和 ConvNeXt token soft contrastive loss；`assets.mindeye1_vae`、`assets.mindeye1_convnext` 指向已有冻结权重。每个完整 epoch 覆盖全部训练刺激，验证 loss 选择低层 best checkpoint。随后将配置中的 `autoencoder_checkpoint_path` 指向此低层检查点，用独立 `predict.py` 命令加载高层检查点执行 img2img。训练阶段不生成图像或运行 benchmark。

MindEye1/LDM 训练指定的单个被试。MindEye2 配方使用其他被试预训练，再在 `--subject` 指定被试的训练集上微调；预测默认选明确记录的微调检查点。MindBridge 配方联合训练。默认参与者集合为 complex 九人、simple 七人（simple 没有 sub-02/sub-03）；二者默认只输出所指定被试的预测，保持文章对应的训练模式。

MindEye2 多卡训练用 `torchrun --standalone --nproc_per_node=<GPU数> train.py ...`。`per_subject_batch_size` 是全局每被试 batch，建议 GPU 数能整除它。样本按刺激 ID 在 tar 内分配到 rank；对比损失使用可反向传播的跨卡 embeddings，保留完整全局负样本集合。MixCo 的被试/卡间索引均加入累计偏移，映射到自身的样本保留完整目标概率。这些修复可能导致结果与旧代码不同。

MindEye2 完整生成使用 `config.mindeye2_full.yaml`，填入已有 GIT、转换器、SDXL、unCLIP、VAE 和 OpenCLIP 权重路径，再运行 `python predict.py --model mindeye2 --subject 1 --stimulus complex --config config.mindeye2_full.yaml --checkpoint <已训练检查点> --run-name mindeye2_full`。完整路线生成 caption，对 unCLIP 图像做 SDXL refinement，再以 0.8 refined + 0.2 blurry 重建进行融合。完整输出的 manifest 应全部记录 `route: unclip_sdxl_refine_lowlevel_blend`；同时保存 refined 和 blurry 图像。`config.yaml` 中 `caption_image_refinement: false` 使用高层 unCLIP 结果，低层图像单独保存。

`models/` 保留四种方法各自的实现；`vendor/` 保留实际运行需要的上游源码。公共评分函数在 `evaluation/`，导入它们不加载 Torch、Transformers 或 baseline。

## 直接调用指标

```python
from avid_benchmark.evaluation import compute_clip_t_2way, compute_token_f1

score = compute_clip_t_2way(image_embeddings, text_embeddings, sample_ids=stimulus_ids)
print(score.score, score.n, score.values)
caption_score = compute_token_f1(predicted_captions, reference_captions, sample_ids=stimulus_ids)
```

每个指标一个独立函数，返回分数、样本 ID、逐样本值、N 和协议版本。矩阵指标使用某一被试的完整固定候选集；多个被试分别计算，不能拼接重复刺激作为负例。Encoder/VQA 抽取与计分分离，CLI 在一次评估中复用模型和 embeddings。

| 类别 | 独立指标 |
|---|---|
| 图像 | PixCorr、SSIM、Inception 2-way、CLIP-I 2-way、AlexNet(2)/(5) 2-way、EfficientNet/SwAV correlation distance |
| 图文语义 | CLIP-T、SigLIP、AltCLIP 2-way；CLIP-T R@1、R@5；固定题库 TIFA |
| 属性 | color、position、visual identity 的 raw accuracy、balanced accuracy、balanced chance |
| Caption | BLEU-1、BLEU-4、ROUGE-L F1、Token-F1、exact match |

数值输入也可通过 `evaluate.py --inputs <arrays.npz> --metrics ... --output ...` 计算。NPZ 中的数组名对应函数参数，例如 `image_embeddings`、`text_embeddings`、`sample_ids`。函数签名和协议说明见 [METRICS.md](METRICS.md)。属性答案使用 `--answers answers.jsonl`，无需重新运行 Qwen。仅对数值或已有答案计分时无需配置文件；抽取图像特征时提供模型路径配置。

预测 manifest 每行至少包含 `subject_id`、`stimulus_type`、`split`、`sample_id`、`image_name`、`generated_image_path`、`gt_image_path`。语义指标还需 reference caption，可直接提供 `reference_caption`，或从配置中的 `used_stimuli.tsv` 读取 `translated_caption`；与模型是否生成 caption 无关。图片路径可以相对 manifest。

## 复现协议

LDM 和 MindBridge 保留历史方法使用真实参考图像选择多个生成候选的协议（分别为 5/16 candidates）；MindEye1 使用脑预测 embedding 选择。候选数和选择方式应与报告一起保存。不要把不同选择协议的结果混为同一实验。

Caption BLEU 保留历史实现的 add-one smoothing：短句/零词重叠仍可能有正值，并非标准 unsmoothed BLEU。空预测计零并进入完整分母；参考文本不能为空。属性固定使用 `qwen_v3`，保留多答案颜色标注，balanced chance 与计分类别分组一致。

## 验证与发布状态

`python -m pytest tests -q` 检查评分和模型修复；`torchrun --standalone --nproc_per_node=2 tests/ddp_gradient_check.py` 验证实际 MindEye2 forward 的两 rank 梯度等于合并 batch 参考，并验证更新后参数一致。

该检查同时验证全局 MixCo/soft contrastive loss 的梯度。测试使用小型 backbone/prior；真实 baseline 的训练和预测仍需单独的 GPU 任务验证。

数据和预训练权重需单独准备，仓库不包含实验记录、结果、集群配置或机器上的实际文件地址。第三方来源及许可说明见 [THIRD_PARTY.md](THIRD_PARTY.md)。TIFA 的干净环境可移植安装仍待验证。

数据目录由 `paths.dataset_root` 指定，可以继续指向已有目录，无需改名或搬动旧数据。配置中的 AVID 路径仅为示例。
