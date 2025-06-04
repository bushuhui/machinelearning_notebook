# 航空航天工程中的机器学习

在航空航天领域，机器学习有着广泛且深入的应用。在设计优化方面，可借助神经网络代理模型优化飞行器气动外形（如波音公司优化机翼设计使燃油效率提升 12%、研发周期缩短 30%），利用遗传算法与强化学习实现结构轻量化（NASA 优化卫星结构减重 15%），还能通过深度学习预测材料性能以加速新型材料研发（如高温合金研发周期从 5 年缩至 18 个月）。故障预测与健康管理领域，LSTM 网络分析卫星遥测数据可使航天器在轨故障预测准确率达 98%，随机森林算法用于航空发动机维护能提前 200 小时预测叶片裂纹、降低 35% 维修成本。自主控制系统中，强化学习助力无人机实现复杂环境避障，Q - learning 优化航天器轨道规划可降低 22% 燃料消耗，深度学习模型让飞行器控制响应速度提升至毫秒级。运营效率提升方面，ML 分析数据能为航线优化节省燃油（空客单次洲际航班省 1.2 吨）、提升机场航班准点率（北京首都机场提升 18%），计算机视觉结合 CNN 检测飞机蒙皮缺陷可使漏检率降至 0.01%。这些应用成果显著，不仅提高了航空航天领域的设计效率、性能和安全性，降低了维护成本与任务风险，还推动了智能化、自动化发展，为行业的高效运营和可持续发展提供了有力支撑。



## 一、机器学习核心技术框架

机器学习（ML）作为人工智能的核心分支，通过数据驱动的方式让系统自动优化性能。其技术体系包括：

* 监督学习（如分类、回归）：利用带标签数据训练模型，实现精准预测[12]。在航空航天中适用于故障诊断、轨迹预测等场景。
* 无监督学习（如聚类、降维）：处理无标签数据，用于异常检测或数据压缩[23]。
* 强化学习（RL）：通过环境交互学习最优策略，是自主导航系统的核心技术[15]。
* 深度学习（DL）：基于多层神经网络处理复杂模式，在图像识别、自然语言处理中表现突出[22]。关键算法包括CNN（图像处理）、LSTM（时序预测）等。

技术实现依赖三大要素：数据质量（需特征工程优化）、模型选择（如随机森林、神经网络）、算法优化（梯度下降等）[21]。



## 二、航空航天工程的主要研究领域

### 2.1 飞行器设计与优化

空气动力学领域，通过神经网络代理模型加速 CFD 仿真优化机翼等气动设计，深度神经网络解决跨马赫数复杂流动建模难题；结构与材料工程领域，借助机器学习 / 深度学习模拟材料参数优化航天器结构，结合强化学习、遗传算法实现轻量化，通过多学科优化整合多领域知识提升综合性能，并优化智能材料与结构设计；热管理与声学领域，数字孪生技术结合机器学习构建虚拟仿真模型，支撑设计优化与航空发动机维护；航天动力学与推进系统虽未明确提及直接应用，但可延伸至轨道优化等潜在场景。


#### （1）空气动力学与气动设计优化  
核心领域：流体力学分析与飞行器外形优化  
- 基础研究范畴  
  - 研究物体（如机翼、涡轮叶片）周围流体流动特性，涉及风洞实验、升力原理及气动特性分析[38] [45]。  
- 机器学习应用  
  - 气动设计优化  
    - 利用神经网络（NN）构建代理模型，加速计算流体动力学（CFD）仿真，优化机翼形状、翼型几何及无人机/发动机叶片设计[112]。  
    - 案例：波音公司通过机器学习自动优化机翼设计，提升气动性能与燃油效率[81]。  
  - 复杂流动建模  
    - 深度神经网络（DNN）应用于跨马赫数的非稳态气动和气弹性建模，解决传统方法难以处理的复杂流动问题[99]。  


#### （2）推进系统与航天动力学  
核心领域：动力供给与轨道规划  
- 基础研究范畴  
  - 推进系统：涵盖内燃机、喷气发动机、火箭等动力装置设计，解决太空/大气环境下的能量供给问题[38]。  
  - 航天动力学：研究轨道设计、优化、多体动力学及空间态势感知，支撑航天器轨道规划与任务执行[40]。  
- 机器学习关联应用  
  - 未在原始条目中明确提及机器学习直接应用，可结合文档其他部分延伸（如轨道优化中的强化学习应用[67]，但当前条目未涉及，故暂不展开）。  


#### （3）结构与材料工程  
核心领域：轻量化设计与材料性能优化  
- 基础研究范畴  
  - 设计航天器结构以承受飞行载荷，聚焦轻量化与安全性平衡，涉及材料选择、强度分析[38]。  
- 机器学习应用  
  - 结构设计与优化  
    - 通过机器学习/深度学习模拟材料参数与结构配置，优化航天器（卫星、火箭等）结构，提升设计效率与性能[67]。  
    - 强化学习+遗传算法结合，实现组件轻量化（如减少重量不影响强度），卷积神经网络（CNN）/生成对抗网络（GAN）用于结构预测与生成[113]。  
  - 多学科优化设计  
    - 融合结构力学、热力学、材料科学等多学科知识，通过机器学习实现综合优化（如速度、载荷、航程等多目标平衡）[116]。  
    - 多任务学习与迁移学习技术提升跨学科设计效率[98]。  
  - 智能材料与结构  
    - 机器学习优化智能材料性能（如变体飞行器自适应结构），增强可靠性与环境适应性[99]。  


#### （4）热管理、声学与数字孪生  
核心领域：环境适应性与仿真技术  
- 基础研究范畴  
  - 热管理：解决飞行器高温环境下的散热与涂层保护；声学：研究噪声控制、声学传播及气动平衡[45]。  
- 机器学习应用  
  - 数字孪生技术  
    - 结合机器学习与大数据，构建飞行器虚拟仿真模型，整合历史数据实现设计优化与故障预测（如航空发动机维护）[104] [99]。  





### 2.2 飞行器制造


在飞行器制造领域，机器学习贯穿设计、材料、制造及测试全流程：总体设计中，通过多学科优化算法整合多系统数据生成最优方案，利用生成对抗网络辅助概念设计；结构设计上，借助强化学习与遗传算法优化拓扑实现轻量化，通过卷积神经网络预测结构疲劳风险；材料科学领域，深度学习模型加速新型材料研发，模拟分子结构预测性能，并优化智能材料参数；制造流程中，实时分析传感器数据优化装配调度，数字孪生技术识别工艺瓶颈，计算机视觉与异常检测算法提升质量控制精度，机器学习还赋能3D打印优化打印路径；测试与验证环节，代理模型减少物理试验次数，虚拟测试平台通过强化学习模拟极端工况验证稳定性。


#### （1）飞行器设计与制造全流程框架  
核心范畴：涵盖从概念设计到生产验证的全周期技术环节。  
- 总体设计  
  - 定义：统筹飞行器性能指标（如航程、载荷、速度），协调气动、结构、动力等多系统集成[56]。  
  - 机器学习应用（补充）：  
    - 通过多学科优化（MDO）算法整合气动、结构、控制等学科数据，生成最优总体方案[116]。  
    - 生成对抗网络（GANs）辅助概念设计，自动探索非常规外形（如仿生机翼）[113]。  

- 结构设计  
  - 定义：设计飞行器承重框架，平衡强度、轻量化与成本[56]。  
  - 机器学习应用：  
    - 强化学习+遗传算法优化结构拓扑，减少组件重量同时保持强度[113]。  
    - 卷积神经网络（CNN）分析应力分布数据，预测结构疲劳风险[99]。  


#### （2）材料科学与创新  
核心目标：开发高性能材料，提升飞行器环境适应性。  
- 基础研究范畴  
  - 研究航空航天结构材料（如钛合金、复合材料）的力学、热学特性，研发新型材料（如形状记忆合金、纳米材料）[50]。  
- 机器学习应用  
  - 材料性能预测  
    - 深度学习模型分析材料成分-工艺-性能关系，加速新型材料研发（如高温合金研发周期从5年缩至18个月）[56]。  
    - 案例：通过图神经网络（GNN）模拟分子结构，预测复合材料抗冲击性能[99]。  
  - 智能材料设计  
    - 机器学习优化智能材料参数（如压电材料响应特性），用于变体飞行器自适应结构[99]。  


#### （3）制造流程优化与质量控制  
核心环节：提升生产效率，确保工艺稳定性与产品可靠性。  
- 制造流程优化  
  - 机器学习应用：  
    - 实时分析生产线传感器数据，优化装配顺序与设备调度，减少工时浪费（如某飞机制造商装配效率提升20%）[74]。  
    - 数字孪生技术模拟制造过程，提前识别工艺瓶颈（如焊接变形风险）[104]。  
- 质量控制  
  - 传统方法：人工检测、无损探伤（如超声波检测裂纹）。  
  - 机器学习升级：  
    - 计算机视觉（CNN）检测零件表面缺陷，漏检率从5%降至0.01%[74]。  
    - 传感器数据实时分析：通过异常检测算法（如孤立森林）识别加工过程中的参数偏移，避免批量缺陷[74]。  
- 工艺创新（补充）  
  - 3D打印（增材制造）：机器学习优化打印路径与材料沉积，减少支撑结构并提升成型精度[89]。  


#### （4）测试与验证  
核心作用：验证设计指标，确保飞行器符合安全与性能要求。  
- 传统测试：风洞试验、静力测试、飞行测试[56]。  
- 机器学习赋能  
  - 代理模型替代部分物理试验：如用神经网络加速CFD仿真，减少风洞测试次数[112]。  
  - 虚拟测试平台：通过强化学习模拟极端工况（如强风切变），测试飞行器稳定性[92]。  





### 2.3 飞行控制

在飞行器控制与飞行动力学领域，机器学习通过数据驱动技术优化全流程：控制理论与系统设计中，运用最优控制、统计学习等理论实现航天器姿态控制与自适应系统设计；飞行控制与导航系统优化上，借助机器学习分析历史数据提升控制算法精度，利用深度学习动态调整策略，强化学习则用于无人机避障及航天器轨道优化；飞行路径与任务规划中，基于历史数据、气象及导航数据生成最优航线，减少燃油消耗，ADSS 系统实时优化航班计划并预警拥堵；测试与数据应用环节，通过机器学习分析飞行测试数据验证设计，利用卫星图像处理技术辅助环境监测与决策。

#### （1）控制理论与系统设计  
核心范畴：基于机器学习的数据驱动控制技术，优化飞行器操纵与轨迹规划。  
- 系统控制理论应用  
  - 涵盖最优控制、非线性控制、统计学习与概率控制理论，用于航天器姿态控制、轨迹生成及控制系统设计[36]。  
  - 数据驱动控制：通过系统辨识（如传感器数据建模）实现自适应控制，提升复杂环境下的稳定性[36]。  

#### （2）飞行控制与导航系统优化  
核心目标：通过机器学习提升控制算法精度，实现自主化、智能化导航。  
* 飞行控制系统优化  
   - 机器学习学习历史飞行数据，优化控制输入（如自动驾驶系统参数），提升复杂工况下的控制准确性与适应性[73] [111]。  
   - 深度学习技术增强系统灵活性，基于实时数据动态调整控制策略（如应对气流扰动时的毫秒级响应）[111]。  
   - 强化学习应用：优化自主导航策略，通过历史数据与实时反馈实现路径规划（如无人机避障、航天器轨道优化）[92] [73]。  

* 自主导航与避障  
   - 太空探索：利用机器学习预测航天器轨迹，结合深度学习识别太空中的物体与危险因素（如陨石、空间碎片），实现自主避障与路径调整[67]。  
   - 无人机系统：通过机器学习实现导航、避障及复杂任务执行（如城市环境自主飞行），确保多场景安全运行[73] [67]。  

#### （3）飞行路径与任务规划  
核心价值：数据驱动的路径优化，降低能耗并提升任务效率。  
* 飞行路径优化  
   - 分析历史飞行数据、实时气象及卫星导航数据，生成最优航线建议，减少燃油消耗与碳排放（如空客单次洲际航班节省燃油1.2吨）[83] [143] [97]。  
   - 深度神经网络预测飞行轨迹，结合无人机模拟器验证方案可行性，降低实际试飞风险[91]。  

* 智能决策与任务支持  
   - 自动化决策支持系统（ADSS）：集成机器学习算法处理航班信息、天气、机场容量等数据，实时优化航班计划，识别交通拥堵点并预警[143]。  
   - 战场环境应用：机器学习助力飞行器感知战场态势，优化战术决策（如自主目标识别与规避）[99]。  

#### （4）测试与数据应用  
核心环节：飞行测试的数据驱动分析与辅助决策。  
- 飞行测试与数据整合  
  - 设计飞行测试程序，通过机器学习分析性能数据（如操纵特性、载荷响应），验证设计目标并支持认证[50]。  
  - 卫星图像处理：利用机器学习自动识别天气系统、监测环境变化（如海洋/陆地动态），为资源管理与任务规划提供数据支持[71]。  



### 2.4 飞行运维


在航空航天维护与运营领域，机器学习通过数据驱动技术实现多维度优化：维护与故障管理中，基于历史和实时传感器数据识别潜在故障，提前规划维护以减少停机成本，如NASA智能分析技术提升故障诊断效率，飞机发动机故障预测可提前200小时预警；空中交通与运营管理中，利用深度学习预测航班流量、优化航线，北京首都机场准点率提升18%，并通过数据可视化辅助决策；能源与环境管理中，优化发动机参数及飞行路径，空客单次洲际航班节省燃油1.2吨，同时助力机场能耗管控与低碳运行；乘客与服务管理中，分析乘客行为数据提供个性化服务，提升航空出行体验。


#### （1）维护与故障管理  
核心目标：通过数据驱动实现设备状态监测与预防性维护，提升可靠性并降低成本。  
* 预测性维护  
   - 通过分析历史/实时传感器数据，识别潜在故障模式，提前规划维护计划，减少非计划停机和维修成本[62] [74] [66] [111]。  
   - 应用场景：飞机发动机组件故障预测（如提前200小时检测叶片裂纹）、航天器设备寿命评估[67] [81] [97] [149]。  
   - 技术价值：避免昂贵的突发故障，优化维护资源分配，提升系统可用性。  

* 故障诊断与检测  
   - 实时监测系统状态，结合计算机视觉、自然语言处理等技术分析遥感数据，实现故障定位与异常识别[81] [97] [149]。  
   - 案例：NASA利用智能分析技术支持飞行器故障检测，提升诊断效率[81]。  


#### （2）空中交通与运营管理  
核心目标：优化航空流量、提升运行效率与安全性。  
* 空中交通与航班管理  
   - 流量预测与优化：通过深度学习分析历史航班、气象等数据，预测流量分布，自动调整航班计划，优化航线和机场资源分配[70] [143]。  
     - 案例：预测特定区域航班密度，辅助空管部门科学决策，北京首都机场航班准点率提升18%[143]。  
   - 延误预测与应对：识别天气、机场运营等延误因素，建立离港预测模型，提前干预以减少延误[147] [150]。  

* 决策支持与数据可视化  
   - 构建分布式数据平台整合航空数据，利用可视化技术将复杂数据转化为图表，辅助管理人员快速决策[143]。  


#### （3）能源与环境管理  
核心目标：降低能耗、减少碳排放，推动绿色航空。  
* 能源效率优化  
   - 调整发动机推力、燃油混合比等参数，优化能源管理系统，提升燃油效率[66] [104]。  
   - 应用场景：飞行路径优化（如空客单次洲际航班节省燃油1.2吨）、机场能耗预测与管控[161]。  

* 环境保护  
   - 监测机场环境数据，通过优化飞行路径减少碳排放，助力绿色机场建设[161]。  


#### （4）乘客与服务管理  
核心目标：提升乘客体验，挖掘数据价值。  
- 乘客行为分析与个性化服务：通过分析乘客数据（如偏好、出行模式），提供个性化航班推荐、酒店预订等服务，优化客户体验并驱动精准营销[160]。  



## 三、机器学习在航空航天领域面临的挑战与限制

机器学习在航空航天领域面临的挑战与限制主要包括以下几个方面：

1. 数据依赖性与质量：机器学习模型的性能高度依赖于训练数据的质量和数量。然而，航空航天领域的数据获取往往受限，且数据稀疏，尤其是在关键决策区域，这导致模型难以有效训练和泛化[63] [171] [176]。此外，数据标注和处理的复杂性也增加了训练难度[173] [184]。
2. 模型可解释性与安全性：在安全关键应用中，如飞行控制和故障检测，机器学习模型的“黑箱”特性是一个重大挑战。模型的决策过程难以解释，这限制了其在航空航天领域的广泛应用[167] [178]。同时，模型的安全性问题也不容忽视，例如对抗性攻击和模型稳定性问题[172] [180]。
3. 认证与合规性：航空航天领域对系统的安全性和可靠性要求极高，现有的认证标准和流程难以适应机器学习技术的非确定性和概率行为。因此，需要开发新的认证方法和标准，以确保AI系统的可信度和安全性[63]。
4. 计算资源与能耗：无人机和航天器通常面临有限的计算资源和能源需求。机器学习算法的高计算需求可能导致能源消耗增加，从而影响任务的持续性和效率[79] [171]。
5. 迁移学习与泛化能力：尽管迁移学习可以减少数据需求和训练时间，但其在跨领域应用中的性能下降和隐私问题仍然是一个挑战[63] [184]。此外，模型在训练数据之外的场景中表现不佳，也限制了其在复杂环境中的应用[171] [176]。
6. 监管与法律挑战：随着机器学习在航空航天领域的应用日益广泛，如何制定相应的法律法规以确保其安全性和合规性成为一个重要问题。例如，自主武器系统的使用是否符合伦理道德，以及如何规范其使用等[90] [180]。
7. 仿真与物理系统的偏差：在航空航天领域，仿真模型与物理样机之间的迁移偏差问题依然存在。过于精确的建模可能导致实时性不足，从而影响模型的准确性[167] [170]。
8. 算法收敛性与学习效率：许多机器学习算法在实际应用中难以收敛，尤其是在处理复杂任务时，如飞行器的制导和控制。此外，学习效率较低也限制了其在实时系统中的应用[167] [79]。
9. 非确定性与鲁棒性：某些机器学习算法的结果可能因相同输入和内部条件而变化，这使得验证和测试变得困难。此外，算法的鲁棒性也是一个重要问题，尤其是在不确定的环境中[63]。
10. 伦理与法律问题：在军事航天领域，人工智能和机器学习的应用还涉及伦理和法律问题，如自主武器系统的使用是否符合伦理道德，以及如何规范其使用等[180]。

机器学习在航空航天领域虽然展现出巨大的潜力，但其在数据依赖性、模型可解释性、认证与合规性、计算资源、迁移学习、算法收敛性、非确定性、鲁棒性、伦理与法律等方面仍面临诸多挑战。这些挑战需要通过技术创新、政策制定和跨学科合作来逐步解决。



## 四、关键技术挑战与应对

1. 数据瓶颈
    * 问题：航天领域试验数据稀缺（如超音速流场数据）
    * 解决方案：迁移学习（复用风洞数据）+生成对抗网络（GANs）合成数据[171]
2. 模型可解释性
    * 问题：深度学习黑箱特性阻碍安全认证
    * 进展：SHAP值分析提供决策依据[63]
3. 实时性约束
    * 案例：立方星采用模型压缩技术，FDIR算法在200MHz处理器上运行[133]
4. 多物理场耦合
    * 创新：图神经网络（GNN）处理结构-热-流体耦合问题[119]



## 五、未来发展方向

1. 数字孪生体： 基于ML的飞行器全生命周期动态仿真[81]
2. 在轨学习系统： 星载边缘计算设备实现模型自主更新[167]
3. 多智能体协同: 集群航天器通过联邦学习共享知识[73]
4. 量子机器学习: 解决超大规模优化问题（如全球航线网络规划）



## 六、结论

机器学习正在重构航空航天工程范式：在设计端实现多目标优化，在制造端提升质效，在运维端构建预测能力，在控制端赋能自主决策。尽管面临数据、算力、认证等挑战，但随着ML与物理模型的深度融合（如PINNs物理信息神经网络），航空航天系统将向更智能、可靠、高效的方向演进。此技术转型要求工程师兼具领域知识与数据科学能力，推动产学研协同创新[73]。



## 参考资料

[1. A. Krizhevsky, I. Sutskever et al. “ImageNet classification with deep convolutional neural networks.” Communications of the ACM](https://doi.org/10.1145/3065386)

[2. Sepp Hochreiter, J. Schmidhuber. “Long Short-Term Memory.” Neural Computation](https://doi.org/10.1162/neco.1997.9.8.1735)

[3. 《机器学习原理及应用》](http://jxky.bcpl.edu.cn/docs/2025-03/0ab0ad30b8a543ee8d582cfc305cd485.pdf)

[4. Foundations of Machine Learning](https://vemu.org/uploads/lecture_notes/18_01_2024_1087039675.pdf)

[5. 机器学习方法](https://www.tup.tsinghua.edu.cn/upload/books/yz/093532-01.pdf)

[6. PRINCIPLES OF MACHINE LEARNING](https://www.cs.utexas.edu/~beasley/syllabus/2025%20Spring%20Machine%20Learning%20Syllabus.pdf)

[7. 机器学习研究进展](https://see.xidian.edu.cn/faculty/xbgao/neuralfuzzy/wangjue.ppt)

[8. 人工智能核心技术综述 [2023-06-28]](https://mp.weixin.qq.com/s?__biz=MzU5OTQ3MzcwMQ%3D%3D&mid=2247518920&idx=1&sn=90c2c05359ec13dbdc8d7a63fbff7ea3&chksm=feb6a182c9c12894e78890cfa0b6f65eda18cd538344fcd8fa101f1e184d7dff4276a193e635&scene=27)

[9. Machine Learning](https://cdn.chools.in/DIG_LIB/E-Book/M1-Machine-Learning-Tom-Mitchell_.pdf)

[10. T. Baltrušaitis, Chaitanya Ahuja et al. “Multimodal Machine Learning: A Survey and Taxonomy.” IEEE Transactions on Pattern Analysis and Machine Intelligence](https://doi.org/10.1109/TPAMI.2018.2798607)

[11. 机器学习有哪些基本方法？](http://aigraph.cslt.org/ai100/pdf/AI-100-34-%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E6%9C%89%E5%93%AA%E4%BA%9B%E4%B8%BB%E8%A6%81%E6%96%B9%E6%B3%95.pdf)

[12. 人工智能的核心概念、发展历程与未来趋势 [2025-03-28]](https://www.360doc.cn/article/16295112_1150045900.html)

[13. 机器学习技术在提升机器人流程自动化中的应用](http://4dpublishinggroup.com/Public/Uploadfiles/20250120/77fad871e3af8910d9c3d8327a24fc99.pdf)

[14. 合肥市国家新一代人工智能创新发展试验区“十四五”规划（2021-2025年）（征求意见稿）](http://yiqi-oss.oss-cn-hangzhou.aliyuncs.com/aliyun/Ueditor/Manage/2021-04-12/20210412-1956394343.docx)

[15. Machine Learning: Algorithms, Real-World Applications and Research Directions](https://link.springer.com/content/pdf/10.1007/s42979-021-00592-x.pdf)

[16. How Can I Teach My Machine to Learn?](https://data.militaryembedded.com/uploads/articles/whitepapers/9178.pdf)

[17. 机器学习:人工智能领域的核心技术 [2024-07-27]](https://www.dongaigc.com/a/machine-learning-core-technology)

[18. 人工智能概述](https://www.tup.com.cn/upload/books/yz/088094-01.pdf)

[19. 机器学习轻量化加速的五大核心技术突破 [2025-06-03]](https://www.51cto.com/aigc/5840.html)

[20. 人工智能技术及其应用](http://www.lamda.nju.edu.cn/lixc/presentations/20200407-AIV.pdf)

[21. 人工智能导论：机器学习基础与应用 [2024-05-14]](https://chatgpttb.cn/article-269937.html)

[22. 图解 72 个机器学习基础知识点 [2024-02-18]](https://zhuanlan.zhihu.com/p/682585565)

[23. 大数据架构师必知必会系列：数据挖掘与机器学习 [2023-12-03]](https://juejin.cn/post/7308219554522202149)

[24. 机器学习和数据挖掘 [2024-06-19]](https://www.ai-indeed.com/encyclopedia/9368.html)

[25. 免费开始你的AI之旅 [2023-08-01]](https://aibus.ai/zh-hans/blogs/aijishudehexinsuanfahefangfayounaxie)

[26. 机器学习 [2025-03-02]](https://oldbird.run/ai/ml/)

[27. 机器学习的核心原理是什么？ [1997-01-01]](https://www.ym163.com/?free-domain-email/4516.html)

[28. Machine Learning Techniques and Applications [2022-08-31]](https://kryptomind.com/https-kryptomind-com-three-essential-ed-to-know-about/)

[29. 人工智能核心技术：深入探讨AI的技术基础 [2024-03-20]](https://www.wulian6.com/a/202308295780.html)

[30. 机器学习的基本原理和方法是什么？ [2023-07-26]](https://cloud.tencent.com/developer/techpedia/1501/13190)

[31. AEROSPACE ENGINEERING](https://openapi.aiu.edu/submissions/profiles/UD87281SP96502/Assignments/a9UD87281_849749_assignment.aerospace.engineering.docx)

[32. 专业解析 | 航空航天工程专业课程设置及研究方向 [2024-10-07]](https://mp.weixin.qq.com/s?__biz=MzA3ODM4MTczMA%3D%3D&mid=2652326065&idx=4&sn=7cb3b7217838439bed4144b4179ed3c1&chksm=8538c777e0ae1fe7654d0b254325e22e5decd6693eebcc44611aa1da7db5bcdbc1785de0ad43&scene=27)

[33. 我国高等学校在航空航天领域的研究现状及前沿分析](https://www.sciengine.com/doi/pdfView/B4C5F6FA35F1473A96C49B8034990CB3)

[34. 北京理工大学全日制硕士专业学位研究生培养方案](https://ls.bit.edu.cn/docs/2016-05/20160530020054789099.pdf)

[35. 航空航天工程专业介绍与就业前景 [2024-12-13]](https://www.51liuxue.com/mobilep/2940)

[36. 航空宇宙工学専攻](https://www.t.kyoto-u.ac.jp/en/admissions/graduate/exam1/doctorsecond2024_e/206_822a7a7a5b875b99-535a58eb30fb51ac-je.pdf)

[37. 基于 Web of Science 的中国航空航天技术发展态势研究](https://journals.istic.ac.cn/qbgc/ch/reader/create_pdf.aspx?file_no=202002010&flag=1&journal_id=qbgc&year_id=2020)

[38. Aerospace Engineering [2024-07-29]](https://engineering.fandom.com/wiki/Aerospace_engineering)

[39. Pushing Boundaries: The Ripple Effect of Aerospace Engineering on Society](https://www.longdom.org/open-access-pdfs/pushing-boundaries-the-ripple-effect-of-aerospace-engineering-on-society.pdf)

[40. AAE Research Areas [2024-01-01]](https://engineering.purdue.edu/AAE/research)

[41. 航空航天工程认证计划 [2022-01-01]](https://accreditation.us/zh-CN/info/education/aerospace-engineering-accreditation-program/)

[42. 航空航天学院](https://m.juyingonline.com/upload/201809/27/201809271438319298.docx)

[43. 科研项目 | STEM专业学子必备！航工航天工程名校科研重磅推荐！ [2024-03-11]](https://www.51liuxue.com/p/2940)

[44. 航空航天包括哪些专业-航空航天类专业目录及专业代码 [2024-04-26]](https://www.dxsbb.com/news/?39432.html)

[45. AERONAUTICAL ENGINEERING](https://ntrs.nasa.gov/api/citations/19960001624/downloads/19960001624.pdf)

[46. 航空航天专业推荐院校 [2019-12-31]](https://m.jjl.cn/article/462799.html)

[47. 航空航天综合多学科交叉与技术是力学发展的源动力——记首届全国航空航天领域中的力学问题学术研讨会](https://pubs.cstam.org.cn/data/article/lxjz/preview/pdf/j2004-191.pdf)

[48. AERONAUTICAL ENGINEERING A CONTINUING BIBLIOGRAPHY WITH INDEXES (Supplement 168)](https://ntrs.nasa.gov/api/citations/19840005068/downloads/19840005068.pdf)

[49. 美国航空航天工程专业概览](http://m.betteredu.net/view/18946)

[50. Engineering Bookshelf [2017-01-01]](http://engineeringbookshelf.tripod.com/aerospace/index.htm)

[51. 神舟十一号成功牵手天宫二号！航空航天专业留学指南来了！ [2016-10-21]](https://zhuanlan.zhihu.com/p/23100934)

[52. 2021版硕士专业学位研究生培养方案](https://grd.bit.edu.cn/docs/2021-08/e7fcee1315434314b72b810c5503e4f2.pdf)

[53. 航空航天工程概述与就业前景 [2006-01-01]](http://www.qianmu.org/%E8%88%AA%E7%A9%BA%E8%88%AA%E5%A4%A9%E5%B7%A5%E7%A8%8B)

[54. 基于CiteSpaceⅡ的航空航天工程前沿研究 [2009-02-15]](https://faculty.dlut.edu.cn/2007011162/en/lwcg/709064/content/140704.htm)

[55. 美国NC State机械与航空航天工程专业 [2017-04-06]](https://bo.aoji.cn/converge/detailArt-876732-2-2.html)

[56. 关于航空航天有哪些专业 [2024-11-24]](https://lx.hssr.ac.cn/new/13135.html)

[57. 上海交通大学航空航天学院2025年工程管理硕士(MEM)招生简介 [2024-06-19]](https://www.educity.cn/mem/5319533.html)

[58. 航空航天专业的研究领域（机械工程专业） [2024-04-06]](https://zhuanlan.zhihu.com/p/690962903)

[59. 探索航空领域最新动态与技术发展的最新航空知识PPT展示 [2024-12-06]](https://hongshangbearing.com/post/1315.html)

[60. Undergraduate Program in Aerospace Engineering [2008-01-01]](https://www.itb.ac.id/undergraduate-program-in-aerospace-engineering)

[61. USE OF ARTIFICIAL INTELLIGENCE IN SPACE RESEARCH](https://www.irjmets.com/uploadedfiles/paper/issue_4_april_2025/74797/final/fin_irjmets1746390457.pdf)

[62. 航空航天领域的人工智能：重新定义工程 [2025-03-26]](https://visuresolutions.com/zh-CN/%E8%88%AA%E7%A9%BA%E8%88%AA%E5%A4%A9%E4%B8%8E%E5%9B%BD%E9%98%B2/%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD/)

[63. ML meets aerospace: challenges of certifying airborne AI](https://elib.uni-stuttgart.de/server/api/core/bitstreams/b2aa4272-62a9-45e5-aebf-884ab31b3d56/content)

[64. Advanced Computational Methods for Optimizing Mechanical Systems in Modern Engineering Management Practices.](https://ijrpr.com/uploads/V6ISSUE3/IJRPR40901.pdf)

[65. Possible applications of artificial intelligence algorithms in F-16 aircraft](https://sjsutst.polsl.pl/archives/2024/vol123/101_SJSUTST123_2024_Krawczyk_Papis_Bielawski_Rzadkowski.pdf)

[66. Improving aircraft performance using machine learning: A review](https://oa.upm.es/81250/1/10041731.pdf)

[67. Application of Artificial Intelligence Technology in Aerospace Field](https://ojs.s-p.sg/index.php/bdai/article/viewFile/15252/pdf)

[68. Machine Learning & Artificial Intelligence in Aerospace Industry](https://axiscades.com/wp-content/uploads/2023/08/Aerospace-whitepaper-1.pdf)

[69. AI与机器学习在航空航天领域的深度应用与未来展望 [2023-12-16]](https://m.weilianliwan.com/technology/2247.html)

[70. 机器学习算法在航空航天领域的应用 [2024-01-19]](https://www.jjblogs.com/post/1218819)

[71. 机器学习在航空航天领域的应用：探索宇宙的奥秘 [2021-04-28]](https://www.jjblogs.com/post/1152714)

[72. Current AI Technology in Space](https://ntrs.nasa.gov/api/citations/20240001139/downloads/Current%20Technology%20in%20Space%20v4%20Briefing.pdf)

[73. The Role of Artificial Intelligence and Machine Learning in Aerospace Engineering [2023-07-15]](https://newsflowhub.com/the-role-of-artificial-intelligence-and-machine-learning-in-aerospace-engineering-2/)

[74. How Machine Learning Models Can Benefit Aerospace Manufacturing [2020-07-07]](https://acubed.airbus.com/blog/adam/how-machine-learning-models-can-benefit-aerospace-manufacturing/)

[75. Applications of AI and ML for Aerospace and Defense Companies [2023-04-28]](https://www.collimator.ai/post/applications-of-ai-in-aerospace-and-defense)

[76. 航空航天中的机器视觉与深度学习应用 [2024-04-05]](https://www.itxiaonv.com/?p=8193)

[77. Analyse de données et modélisation méso-échelle des nuages de CO2 dans les nuits polaires martiennes](https://theses.hal.science/tel-04075743v1/file/CAILLE_Vincent_these_2023.pdf)

[78. 基于人工智能的工程优化设计与仿真分析 [1997-02-26]](http://xueshu.qikan.com.cn/preview/1/205/4355064)

[79. Machine Learning-Aided Operations and Communications of Unmanned Aerial Vehicles: A Contemporary Survey](http://www.cister.isep.ipp.pt/docs/machine_learning_aided_operations_and_communications_of_unmanned_aerial_vehicles__a_contemporary_survey/1867/attach.pdf)

[80. Application of Multidisciplinary Optimization and Artificial Intelligence Techniques to Aerospace Engineering (Volume II) [2023-11-22]](https://www2.mdpi.com/journal/aerospace/special_issues/JM67056A49)

[81. 人工智能在航空航天设计中的应用与培训 [2024-05-11]](https://www.renrendoc.com/paper/327587524.html)

[82. AI & ML – A Boon to Revolutionize Aerospace [2022-09-28]](https://alliance.edu.in/blog/2022/09/28/ai-ml-a-boon-to-revolutionize-aerospace/)

[83. 机器学习和人工智能在航空航天中的应用 [2024-07-17]](https://www.jindouyun.cn/document/industry/details/180338)

[84. Adrian Carrio, Carlos Sampedro et al. “A Review of Deep Learning Methods and Applications for Unmanned Aerial Vehicles.” J. Sensors](https://doi.org/10.1155/2017/3296874)

[85. 人工智能机器学习的应用场景 [2023-09-07]](https://www.transwarp.cn/keyword-detail/10263-24)

[86. AI技术在航空航天领域的应用与未来展望 [2025-02-12]](https://www.pbids.com/news/1886777822133927936)

[87. 来自人工智能和机器学习在航空航天领域的当前应用和创新 [2022-06-07]](https://www.x-mol.com/paper/1536784379878617088/t?adv)

[88. 数据驱动的航空航天工程：用机器学习重构行业 [2020-08-24]](https://www.x-mol.com/paper/1298676800828968960/t?adv)

[89. 智能制造在航空航天行业中的应用前景 [2024-01-09]](https://max.book118.com/html/2024/0208/6152215152010044.shtm)

[90. Machine Learning in eVTOLs - The Backbone of Autonomous Flight [2021-03-22]](https://www.kdcresource.com/insights/machine-learning-in-evtols-the-backbone-of-autonomous-flight/)

[91. 飞行技术在智能航空器设计中的优化与应用探讨](https://cn.usp-pl.com/index.php/jxgc/article/view/160593/159337)

[92. Air Learning: An End to End Learning Gym for Aerial Robots](https://mlsys.org/media/mlsys-2020/Slides/1370.pdf)

[93. 深度学习在飞行器动力学与控制中的应用研究综述](https://lxsj.cstam.org.cn/en/article/pdf/preview/10.6052/1000-0879-20-077.pdf)

[94. 第二届中国空气动力学大会征文通知（第三轮） [2022-05-18]](https://mp.weixin.qq.com/s?__biz=MzAwNDk2Nzg2NQ%3D%3D&mid=2247486782&idx=1&sn=dc3b3de364655f75c5f055718de50fca&chksm=9b2291e8ac5518feae5434af674e4e964e49a9d9cdae02f3d07950b3020bfc224982bc258ec0&scene=27)

[95. 多模态飞行理论 [2025-01]](https://6565kj.com/Think-Tank/Article/Aerospace%20Engineering/%E5%A4%9A%E6%A8%A1%E6%80%81%E9%A3%9E%E8%A1%8C%E7%90%86%E8%AE%BA)

[96. 一种基于数据驱动与可分离形状张量的代理模型优化设计方法](https://patent-image.qichacha.com/pdf/f4fcc00d4b48c683100d339e52db66b6.pdf)

[97. 机器学习在航空航天工程中的优化和安全性提升 [2023-10-11]](https://www.360doc.cn/article/26181007_1099766629.html)

[98. Transfer optimization in complex engineering design](https://dr.ntu.edu.sg/bitstream/10356/105865/1/AlanTan-PhDThesis_v4.3_signed.pdf)

[99. 飞行器智能设计愿景与关键问题 [2021]](https://hkxb.buaa.edu.cn/EN/10.7527/S1000-6893.2020.24752)

[100. 第二届中国空气动力学大会征文通知（第二轮） [2022-03-31]](https://mp.weixin.qq.com/s?__biz=MzAwNDk2Nzg2NQ%3D%3D&mid=2247486681&idx=1&sn=63d4e0a831dbf86009728e4264aa5b09&chksm=9b22900fac551919823ccc8184d5c34939bc0355bf9acaa058f7e3eb577c5eff0b4758b6050d&scene=27)

[101. 机械结构协同优化设计理论与实践](http://www.issplc.com/upload/pdf/2025/01/21%E6%9C%BA%E6%A2%B0%E7%BB%93%E6%9E%84%E5%8D%8F%E5%90%8C%E4%BC%98%E5%8C%96%E8%AE%BE%E8%AE%A1%E7%90%86%E8%AE%BA%E4%B8%8E%E5%AE%9E%E8%B7%B5.pdf)

[102. 一种气动大差异性数据多任务学习方法](https://pubs.cstam.org.cn/data/article/kqdlxxb/preview/pdf/2021-222.pdf)

[103. 基于机器学习的巡飞弹气动优化与制导一体化设计 [2012-01-02]](https://bzxb.cqut.edu.cn/html/202409/2096-2304%282024%2909-0038-10.html)

[104. 航天总体设计的智慧研发模式 [2024]](https://mp.weixin.qq.com/s?__biz=Mzg5OTg4MTYzNA%3D%3D&mid=2247485505&idx=1&sn=fc4d327a4cf78d2384220be49b0bcc4f&chksm=c1d6e71bb018e956fb31cc27801f211f85757203d4fb1a756292ce01a3d0e889cad7f481a076&scene=27)

[105. Design of a Maneuverable Rocket Using Machine Learning Tuning](https://www.sjsu.edu/ae/docs/project-thesis/William.Miller-F21.pdf)

[106. AI驱动高速飞行器多学科发展知识图谱分析 [2024-06-21]](https://hkxb.buaa.edu.cn/CN/10.7527/S1000-6893.2024.30566)

[107. 大模型在航空航天领域的应用 [2024-02-20]](https://www.dtstack.com/bbs/article/15424)

[108. 南京理工大学专业学位硕士研究生培养方案](https://gs.njust.edu.cn/_upload/article/files/36/a4/ba0975a84d2a8625b73a973d03d8/5b4d5453-1f6c-41b9-bdce-8eacfbcdc6a0.pdf)

[109. 基于数据挖掘的飞行器气动布局设计知识提取 [2020-10-16]](https://hkxb.buaa.edu.cn/CN/10.7527/S1000-6893.2020.24708)

[110. 通过机器学习和更高效的模型改进飞机设计 [2022-09-26]](http://www.cechina.cn/m/article.aspx?ID=74875)

[111. 飞行器自动控制系统的设计与优化 [1997-02-26]](http://www.qikan.com.cn/article/120c2023363341.html)

[112. S. L. Clainche, E. Ferrer et al. “Improving aircraft performance using machine learning: a review.” ArXiv](https://doi.org/10.1016/j.ast.2023.108354)

[113. 登顶Nature！打破传统范式！90年博后再获重大突破，材料领域迎来史诗级进展！ [2024-05-27]](https://mp.weixin.qq.com/s?__biz=MzA4NDk3ODEwNQ%3D%3D&mid=2698886353&idx=1&sn=8fb6d30bd7c5c833600674739dd644f4&chksm=bbaaed1d1f57a9ddfe540dca988b344f8a2bcd6d169fcd7fbce5c1032ed9e9d262d2c6026c74&scene=27)

[114. 东航波音客机坠毁！全球专利超8万件的波音公司成众人集火的靶子！ [2022-03-25]](https://gongkong.ofweek.com/2022-03/ART-310081-8420-30554930_3.html)

[115. 智能未来：人工智能如何改变飞行技术的发展方向？ [2023-10-16]](https://m.chinaflier.com/thread-228530-1-1.html)

[116. 2024年飞行器气动及交叉学科综合设计技术论坛 [2024-10-05]](https://www.huiyi-123.com/article/3259-179.html)

[117. Spacecraft in Orbit Fault Prediction Based on Deep Machine Learning](https://iopscience.iop.org/article/10.1088/1742-6596/1651/1/012107/pdf)

[118. A Machine Learning Approach to Anomaly Detection and Fault Diagnosis for Space Systems [2006]](https://www.doc88.com/p-8708611291280.html)

[119. 空间无人系统智能精准运维: 机制、技术与应用](https://www.sciengine.com/doi/pdfView/BA0E46C88BE0494F9E39FEA8ED8C0DCF)

[120. 人工智能在航天故障诊断中的应用](https://www.jianshu.com/p/aa86abb7be3d)

[121. AI与可持续发展展望](https://www.zhaojunhua.org/reports/AI_SUSTAINABLE_DEVELOPMENT_OUTLOOK_ZH.pdf)

[122. Intelligent Systems and Optimization in Engineering](https://www.isres.org/books/intteligent_system_2024_29-12-2024.pdf)

[123. 空间控制技术与应用 [2023-08-26]](http://journal01.magtech.org.cn/Jwk3_kjkzjs/CN/volumn/volumn_1235.shtml)

[124. Machine Learning Ensures Reliability for Space Missions: To the Moon and Beyond [2019-09-09]](https://www.dataversity.net/machine-learning-ensures-reliability-for-space-missions-to-the-moon-and-beyond/)

[125. NASA SBIR-2024-Phase 1 Solicitation](https://www.nasa.gov/wp-content/uploads/2024/01/sbir-24-i-v2.pdf?emrc=67817da557633)

[126. 航天科技的未来：探索航天器设计与优化的技术进展 [2024-05-14]](https://www.bsvps.com/?id=130989)

[127. Boeing and Saber leverage artificial intelligence to troubleshoot satellites](https://www.boeing.com.au/content/dam/boeing/en-au/pdf/news/2020-11-25-Boeing-and-Saber-leverage-artificial-intelligence-to-troubleshoot-satellites.pdf)

[128. A Failure Diagnosis System Based on a Neural Network Classifier for the Space Shuttle Main Engine](https://ntrs.nasa.gov/api/citations/19900019343/downloads/19900019343.pdf)

[129. Expert-Informed Hierarchical Diagnostics of Multiple Fault Modes of a Spacecraft Propulsion System](https://papers.phmsociety.org/index.php/phmap/article/download/3596/2202)

[130. Anomaly Detection on Partial Point Clouds for the Purpose of Identifying Damage on the Exterior of Spacecrafts](https://ir.lib.uwo.ca/cgi/viewcontent.cgi?params=/context/etd/article/12025/&path_info=Thesis_FinalDraft_2.pdf)

[131. Unreal Engine与EasyV：推动数字孪生在航空航天中的应用 [2025-03-04]](https://easyv.cloud/c/article/14044.html)

[132. 航天智控: 人工智能在故障诊断中的应用 [2024-10-14]](https://www.aigc.cn/33974.html)

[133. 基于机器学习和神经网络的立方体卫星故障检测、隔离与恢复算法开发 [2021-08-23]](https://aerospaceresearch.net/?author=33)

[134. 航天器智能诊断系统开发研究 [1999-06-25]](https://hkxb.buaa.edu.cn/CN/abstract/abstract11310.shtml)

[135. 太空探索中的人工智能：从火星上的自主探测车到人工智能驱动的数据分析 [2024-08-26]](https://julienflorkin.com/zh-CN/%E6%8A%80%E6%9C%AF/%E8%88%AA%E5%A4%A9/ai-in-space-exploration/)

[136. Statler College Media Hub | WVU engineers address NASA problems through artificial intelligence [2022]](https://media.statler.wvu.edu/news/2022/01/24/wvu-engineers-address-nasa-problems-through-artificial-intelligence)

[137. GSoC2021: Development of an FDIR algorithm using Neural networks [2021-08-23]](https://aerospaceresearch.net/?p=2372)

[138. 基于深度学习及GPU计算的航天器故障检测技术 [2020-05-25]](http://jsjclykz.ijournals.cn/jsjclykz/article/html/201910090971)

[139. 太空探索与人工智能：NASA 的 Artemis 计划和 2024 年的人工智能发现 [2024-09-09]](https://www.editverse.com/zh-CN/%E5%A4%AA%E7%A9%BA%E6%8E%A2%E7%B4%A2%E4%BA%BA%E5%B7%A5%E6%99%BA%E8%83%BD/)

[140. 智能航天测控，人工智能的革新引领航天控制领域 [2024-01-01]](https://www.aigc.cn/29861.html)

[141. 人工智能在航天器控制中的应用 [2018-07-10]](http://fkytc.ijournal.cn/html/2018/1/20180102.html)

[142. 数据驱动的航天器故障诊断研究现状及挑战 [2023-02-06]](http://jemi.cnjournals.com/jemi/article/abstract/20210201?st=article_issue)

[143. 基于人工智能的民航空中交通流量管理优化研究](https://cn.sgsci.org/gse/article/download/327/275/1100)

[144. まえがき](https://www.enri.go.jp/jp/about/doc/publication/R2_all.pdf)

[145. 平成30年度業務実績等報告書](https://www.mpat.go.jp/disclosure/source/mpat_H30_all.pdf)

[146. 航班地面保障过程决策支持体系建模](https://www.china-simulation.com/EN/article/downloadArticleFile.do?attachType=PDF&id=3514)

[147. The many ways machine learning has revolutionized the aviation industry [2023-10-16]](https://blog.mysticmediasoft.com/the-many-ways-machine-learning-has-revolutionized-the-aviation-industry/)

[148. Machine Learning and Importance of Data in Aviation [2023-11-15]](https://www.aviationfile.com/machine-learning-in-aviation/)

[149. 机器学习算法在航空预测中的应用研究 [2023-12-11]](https://www.108ai.com/post/9342.html)

[150. 臺灣博碩士論文加值系統 [2025-04-07]](https://ndltd.ncl.edu.tw/cgi-bin/gs32/gsweb.cgi/login?o=dnclcdr&s=id%3D%22104NCKU5294003%22.&searchmode=basic)

[151. 全球航空分析市场按组件（解决方案、服务）、按功能（财务、运营、销售和营销）、按应用（燃油管理、飞行风险管理、客户分析、导航服务）、按最终用途（OEM、售后市场）和地区、2022年至2029年全球趋势和预测 [2024-06-17]](https://exactitudeconsultancy.com/zh-CN/%E6%8A%A5%E5%91%8A/17975/%E8%88%AA%E7%A9%BA%E5%88%86%E6%9E%90%E5%B8%82%E5%9C%BA/)

[152. Artificial Intelligence In Aviation Market Size & Future Growth 2032 [2024-11-30]](https://www.wiseguyreports.com/cn/reports/artificial-intelligence-in-aviation-market)

[153. Artificial Intelligence Roadmap: A human-centric approach to AI in aviation](https://www.easa.europa.eu/sites/default/files/dfu/EASA-AI-Roadmap-v1.0.pdf)

[154. 空中交通管制中的人工智能应用 [2024-04-17]](https://www.qikanchina.com/thesis/view/8179735)

[155. 突破传统--新技术给航企收益管理带来巨大改变 [2023-08-04]](http://www.caacnews.com.cn/1/tbtj_/202308/t20230804_1369443_wap.html)

[156. AGIFORS [2017-01-01]](https://agifors.org/symposium_agenda_2019)

[157. 航空器故障诊断与预测的机器学习算法研究 [2024-09-09]](https://www.lfqiye.com/article/652495771477.html)

[158. 听到你的客户:航空公司案例研究分析 [2023-04-13]](https://www.sobheroshan.com/blog/airline-case-study-analysis/)

[159. 现代国际机场中的机器学习技术，无限可能的探索 [2024-12-08]](https://uzo.hujyw.com/post/1722.html)

[160. 企业AI解决方案中的优化技术 [2020]](https://www.toolify.ai/zh/ai-news-cn/%E4%BC%81%E4%B8%9Aai%E8%A7%A3%E5%86%B3%E6%96%B9%E6%A1%88%E4%B8%AD%E7%9A%84%E4%BC%98%E5%8C%96%E6%8A%80%E6%9C%AF-1796300)

[161. 航空分析市场规模和预测 [2024-11-26]](https://www.marketresearchintellect.com/zh/blog/aviation-analytics-market-the-hidden-power-behind-smarter-aerospace-and-defense-operations/)

[162. 航空数据分析是什么 [2024-05-18]](https://www.vientianeark.cn/qa/240242.html)

[163. 在航空使用AI对风险的发现和解决 [2019-05-13]](http://www.rongyidianshang.com/news-insights/impact-story/using-ai-risk-discovery-and-resolution-aviation)

[164. AVIATION WEEK & SPACE TECHNOLOGY](https://aviationweek.com/sites/default/files/2025-01/AWST_2025_1-13_0.pdf)

[165. 人工智能在空间机器人中的应用现状与关键问题 [2022]](https://mp.weixin.qq.com/s?__biz=MzIyMTYwMDk3MA%3D%3D&mid=2247518858&idx=3&sn=96ab08eac8a6202bae95f08f396cd56d&chksm=e838872fdf4f0e39312630b1e30e7110b27531a07e7690176fc55a3ba92d9d25ad7cf70169dc&scene=27)

[166. Robust concept development utilising artificial intelligence and machine learning](https://oj.sfkvalitet.se/2024ExamensarbeteHAKK.pdf)

[167. Mod-Sim and Decision-Making in Aerospace Domain](https://ntrs.nasa.gov/api/citations/20240007681/downloads/Alexandrov%20slides.pdf)

[168. APPLICATION OF MARKOV CHAINS, MTBF AND MACHINE LEARNING IN AIR TRANSPORT RELIABILITY](https://bibliotekanauki.pl/articles/56402834.pdf)

[169. 基于 LSTM 模型的飞行器智能制导技术研究](https://pubs.cstam.org.cn/data/article/lxxb/preview/pdf/20-388.pdf)

[170. 无人机战争：空中制胜之道](https://microbook.oss-cn-beijing.aliyuncs.com/a4d8e7b43bbd52410d77d725fa32b9c8.pdf)

[171. Hazim Shakhatreh, Ahmad H. Sawalmeh et al. “Unmanned Aerial Vehicles: A Survey on Civil Applications and Key Research Challenges.” ArXiv](https://doi.org/10.1109/ACCESS.2019.2909530)

[172. What is scientific machine learning and how will it change aircraft design? [2020-07-02]](https://www.aerospacetestinginternational.com/opinion/what-is-scientific-machine-learning-and-how-will-it-change-aircraft-design.html)

[173. Towards the Jet Age of Machine Learning [2018-05-02]](https://www.ml.cmu.edu/news/news-archive/2016-2020/2018/may/towards-the-jet-age-of-machine-learning.html)

[174. 深度学习原理与实战：深度学习在航空航天领域的应用 [2023-12-06]](https://zhuanlan.zhihu.com/p/670755073)

[175. Artificial Intelligence Gets Ahead of the Threats [2018]](https://aerospace.org/Annual-Report-2018/artificial-intelligence-gets-ahead-threats)

[176. 机器学习与人工智能在太空探索中的应用 [2023-09-25]](https://t24global.com/space/)

[177. 机器学习赋能：太空态势感知的未来之道 [2024-10-07]](https://www.showapi.com/news/article/670327bb4ddd79f11a336c31)

[178. C4ISR、人工智能与机器学习：军事航天领域的新变革 [2024-12-15]](https://docs.feishu.cn/v/wiki/C6mJwMZpyidcZKkYiSpcOcIhnwb/a2)

[179. MACHINE LEARNING FOR EXTREMELY COMPLEX ENVIRONMENTS [2019-10-07]](https://myventurepad.com/machine-learning-for-extremely-complex-environments/)

[180. S. Brunton, J. Kutz et al. “Data-Driven Aerospace Engineering: Reframing the Industry with Machine Learning.” ArXiv](https://doi.org/10.2514/1.J060131)

[181. 2022年航空业的四大机器学习趋势及其影响 [2022-03-15]](https://datascience.aero/top-4-machine-learning-trends-2022-how-could-affect-aviation/)

[182. D. Baron. “Machine Learning in Astronomy: a practical overview.” arXiv: Instrumentation and Methods for Astrophysics](https://arxiv.org/abs/1904.07248)
