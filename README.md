# GAN_Poem_Generate
## Extend projects of NLP courese which achieve generating Chinese poems by GAN model
### 说明
自然语言处理导论课程的扩展可选项目.实现了根据输入的七言绝句的第一句生成后三句的功能</br>
基本框架是LeakGAN<sup>1</sup>加入输入编码器模型</br>
感谢合作的Y同学在项目中的带飞qwq</br>
### 环境
- **GPU：Nvidia GTX 1080Ti**
- **OS：Ubuntu 16.04**
- **Environments：CUDA9.0,Python3.5**
- **Libs：tensorflow 1.12.0,NLTK 3.3,Numpy 1.14.0**
### 代码实现
- `data`:保存训练、验证和测试数据的目录</br>
- `save`:保存模型checkpoint、预处理理得到的字与ID相互映射的字典、代码运⾏行行过程中临时的⽣生成数据和验证集⽣生成结果等</br>
   - `gen_val`:保存训练过程中在验证集上的⽣生成结果</br>
   - `model`:下分别保存LeakGAN模型训练过程中的最近五个和最佳五个的checkpoint以及预训练过程中效果最好的生成网络checkpoint</br>
   - `word_index_dict.pk`和`index_word_dict.pk`分别为使用`pickle`保存的字典，用于将汉字与对应ID互相映射</br>
- `test-bleu.py`：专⻔编写的测试BLEU值的脚本,调用`nltk`包中方法计算并评估生成诗的优度，要求参考文件和待测文件每行均是完整的4句七言</br>
- `main.py`:训练、测试模型入口，通过`tf.app.flags`封装了命令行参数方便训练和测试，主要参数有以下几个：</br>
   - `-restorn`从最近的一个checkpoint恢复模型，checkpoint路径可通过`-model_path`指明;</br>
   - `-train_data`和`-val_data`分别指明训练和验证数据集；</br>
   - `-infer`读取`test_data`进行生成，否则进行训练;</br>
   - `target_path`生成目标文件路径；</br>
- `utils`：包含了数据预处理的`utils.py`模块和用于生成器、鉴定器训练数据加载以及实际测试数据加载的`DataLoader.py`模块；此外还包含了Texygen<sup>2</sup> 的`metrics`包，其中包含各种文本生成评测指标；
   - `utils.py`包含对数据预处理理获取相关汉字/ID字典、汉字与ID相互转换等函数；
   - `DataLoader.py`实现了`DataLoader`、`TestDataLoader`、`DisDataLoader`三个类，分别对应于生成器训练的数据加载器、测试数据加载器和鉴别器数据加载器，
   每个加载器可以不断生成由数据获取的batch</br>
- `model`包是模型实现的主要部分：
   -`Gan.py`实现了`LeakGan`的基类；<br>
   -`Generator.py`实现`LeakGan`的`Generator`生成器类，修改自Texygen和LeakGAN论文的实现<sup>3</sup>，其中增加了一个全连接层的输入编码层替换原始的
   网络初始条件，以求实现根据输入生成后续内容的要求;</br>
   -`Discriminator.py`实现了`LeakGan`的`Discriminator`鉴别器类
   -`Reward.py`实现了对`LeakGan`进行强化学习训练需要的reward,来自Texygen和LeakGAN论文的实现
   -`LeakGAN.py`实现了`LeakGan`，是模型的主类，在Texygen实现的基础上进行了大量针对任务的修改，实现了针对根据输入生成后续序列这⼀任务的生成器、鉴别器数据加载方法和训练
方法，和对测试数据进行生成的方法，以及训练过程中的指标测试和更新、保存逻辑，⽹络和训练的超参数也在这里修改。 
-其他参数：鉴别器嵌入维度64，dropout_rate为0.75，预训练epoch为80，对抗训练epoch为100，其余参数与Texygen实现中LeakGAN默认值相同</br>
-除上文已提到的地⽅之外，代码思想和具体实现还参考了SeqGAN论文作者的代码<sup>4</sup>以及⼀个采⽤用 SeqGAN进行⽂本⽣成的项⽬Chinese-Hip-pop-Generation<sup>5</sup> </br>
- 运行逻辑：读⼊训练数据与验证数据 -> 处理得到汉字与ID的相互映射字典 -> 据此标注训练数据 -> 预训练生成器Generator -> 
预训练鉴别器器Discriminator -> 对抗训练Adversarial Training并每5个epoch验证一次验证集表现，保存checkpoint -> 训练结束</br>

### 结果 </br>
|Avg BLEU-1|Max BLEU-1|Avg BLEU-2|Max BLEU-2|
|----------|----------|----------|----------|
|0.9658|0.9726|0.6902|0.7035|

模型根据`test.txt`生成的后三句保存在`generated_poem.txt`中，输入一句生成的完整的四句诗保存在`generated_poem_raw.txt`中</br>

### 参考资料
---------------------------------------------
1.Long Text Generation via Adversarial Training with Leaked Information, AAAI 2018, [https://arxiv.org/abs/1709.08624](https://arxiv.org/abs/1709.08624)</br>
2.Texygen: A text generation benchmarking platform, [https://github.com/geek-ai/Texygen](https://github.com/geek-ai/Texygen)</br>
3.Codes of LeakGAN: Text generation using GAN and Hierarchical Reinforcement Learning, [https://github.com/CR-Gjx/LeakGAN](https://github.com/CR-Gjx/LeakGAN)</br>
4.SeqGAN: Implementation of Sequence Generative Adversarial Nets with Policy Gradient, [https://github.com/LantaoYu/SeqGAN/](https://github.com/LantaoYu/SeqGAN/)</br>
5.Chinese-Hip-pop-Generation: Generate Chinese hip-pop lyrics using GAN, [https://github.com/TobiasLee/Chinese-Hip-pop-Generation/](https://github.com/TobiasLee/Chinese-Hip-pop-Generation/)</br>
