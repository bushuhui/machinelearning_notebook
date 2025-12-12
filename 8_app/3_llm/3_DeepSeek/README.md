# DeepSeek

DeepSeek V3 和 R1 是深度求索公司两条并行的模型路线，核心差异可以概括为“通用快” vs “推理深”：

1. DeepSeek-V3  
   定位：通用底座模型，主打高吞吐、低成本。  
   架构：671 B 参数的 MoE，14.8 T token 预训练；支持 16 K 上下文，推理速度≈800 token/s。  
   特点：  
   - 非思考模式，直接输出，延迟低。  
   - 在文本生成、多语言、代码、百科问答等“广度”任务上表现均衡，可类比 GPT-4o 的国产平替。  
   - 最新版 V3.2 支持通过 enable_thinking 开关一键切换“思考”或“非思考”模式，兼顾速度与深度。

2. DeepSeek-R1  
   定位：推理专用模型，主打复杂逻辑和可解释性。  
   架构：在 V3 基础上用强化学习（GRPO）继续训练，参数 670 B；每次回答前强制进行链式思考（Chain-of-Thought），响应更长、速度≈350 token/s。  
   特点：  
   - 数理、编程、长文档分析等“深度”任务准确率比 V3 高 20–40%；FRAMES 长文本问答 82.5 %，ArenaHard 胜率 92 %。  
   - 输出含 reasoning_content 字段，可展示完整思考步骤，方便教学与审计。  
   - 对提示词不敏感，简洁指令即可，避免“角色扮演”式提示干扰推理。






## 参考资料

视频讲解：

* [How DeepSeek Rewrote the Transformer](https://www.youtube.com/watch?v=0VLAoVGf_74)