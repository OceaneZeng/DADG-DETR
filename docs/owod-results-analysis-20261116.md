# 2026-11-16：S路由与M Task1结果分析

## 输入与运行身份

- S附件 `C:/Users/23642/Downloads/consolesowodb.log`，SHA256
  `82c459f49ccf84968d3a27023ff3c3d08a0594a21bfa1fa740651618fe3633c1`。
  main / `2002056e`，v7 hard/soft，Task2各8轮increment、每卡batch1、双3090，
  VOC2007 11点AP50，无balanced finetune。附加只读服务器learning probe见
  [定位对照](sowodb-task2-localization-experiment.md)。
- M附件 `C:/Users/23642/Downloads/train.log`，SHA256
  `9ec4695a436b15cc50034e94b1f860a8083936cca5d2499dfb9eca8866c80b09`。
  mowodb / `f8bc2c64285d083feaede320d6372821cd33028d`，v8 focal、lr5e-5、
  backbone lr5e-6、batch4、distributed=False；目录名为A40全链，但该阶段train.log
  没有GPU型号枚举，不能仅凭目录确认实际显卡。Task1计划50轮，最后完整评估第20轮，
  附件结束在第21轮step3600/4137，无退出/报错证据，不能把截断当成训练结束。
  M用作者VOC积分AP50，manifest SHA256为14de2231…。

## S结论

固定第8轮hard/soft：Previous63.3823/63.2711%，Current2.5996/3.9701%，
Known31.4714/32.1381%，U-Recall1.8925/2.5431%，precision0.8144/1.1579%，
unknown数27505/25996，WI0.042224/0.040326，A-OSE8791/8492。
soft比hard更好，但新类AP与unknown仍低。训练图原始框覆盖26.04/31.10%，
去unknown竞争仅各增加2个GT覆盖；应优先验证定位/表示适配容量。

## M实际进展

Task1无Previous，Current=Known；不是Task2增量新类AP，也不代表完整四任务成绩。
以下为附件实际完成轮次，不按最高测试指标挑选权重。

| 完成轮数 |Current/Known AP50 %|U-Recall %|unknown precision %|unknown数|WI@0.8|A-OSE|
|---|---:|---:|---:|---:|---:|---:|
|8|41.7655|9.6141|0.9390|238764|0.079887|24722|
|12|46.5661|11.5480|0.9734|276664|0.079511|19931|
|16|49.8228|13.3705|0.9666|322570|0.068448|14273|
|18|52.5788|12.2041|0.9035|314990|0.076312|16170|
|20|52.4442|8.2933|0.5646|342551|0.071819|12927|

M已知检测持续学到第18轮附近，不能用S Task2的个位数概括M；但Task2尚无记录，
不能称M增量学习已解决。M第16→20轮正确unknown由3118降至1934，输出多19981，
precision由0.9666%降至0.5646%。更低A-OSE没有抵消召回/precision退化。
高预测量可能含重复、未标注区域或错误类别，不能把所有FP都叫纯背景。
M训练尚未固定末轮，也没有当前方法的同起点完整control对照，不能判断SOTA或最终优劣。

## 优化顺序与实现范围

1. S复用已有soft+bbox-last，独立配对与相同Task1起点；详见定位对照说明。
2. S按现有训练图probe检查框覆盖、分类保留与实际gate；不同时改评分、lr或batch。
3. M已有mowodb分支 `3fde4ee` 的v8/v9可信背景novelty负监督独立对照，
   针对foreground-only novelty；它尚无本次AP结果，不重复实现、不改正在跑的v8链。
   背景误选也可能损失未知召回，必须同时看AP、U-Recall、precision/数量与WI/A-OSE。
4. focal在M有既有同轮证据，但不能直接推广到S。两协议的任务、类别、AP版本、
   方法版本、学习率、有效batch和训练阶段均不同，不作跨协议因果比较。

本地本次仅准备诊断入口和检查，不声称获得优化后的AP；保留已有checkpoint、日志和用户改动。

## 本次验证与证据归档

本地现有tree-detr环境：35项直接相关检查通过，覆盖两种定位策略的真实小型
Deformable DETR CUDA反向、坐标层更新/冻结、旧分类行保持、state重载、配对身份与
跨组恢复拒绝，以及固定第3/8轮评估和随机状态保持。7条提示来自既有torchvision/meshgrid。
1项已知Windows双进程Gloo检查未重跑；本次新增路径尚未在目标Linux双3090上运行。
没有运行新八轮实验，也没有优化后的AP结果。

忽略目录 `output/analysis/owod-20261116/` 保存16份服务器原始JSON及逐文件SHA256
的合并证据 `s-task2-server-evidence.json`，以及20轮M附件指标
`m-task1-attachment-metrics.json`。这些生成证据不加入Git。
