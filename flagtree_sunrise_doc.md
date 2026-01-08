## flagtree 文档

### 一、从Triton迁移代码

#### 1.1 代码梳理
S2的Triton基于3.4版本，对公共部分代码的修改包括：
- SunriseMmaEncodingAttr属性相关的定义和实现
    - `include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td`
    - `lib/Dialect/TritonGPU/IR/Dialect.cpp`
    - `lib/Dialect/TritonGPU/IR/LinearLayoutConversion.cpp`
- `lib/Dialect/TritonGPU/Transforms`内的少部分pass
    - Pipeliner/ScheduleLoops.cpp
    - Coalesce.cpp
    - ReduceDataDuplication.cpp
    - Utility.cpp
- LienarLayout相关
    - `lib/Tools/LinearLayout.cpp`

需要 **【闭源】** 的代码包括：
- `third_party/sunrise/lib`内
    - TritonSunriseGPUToLLVM 目录
    - TritonSunriseGPUTransforms 目录

其他部分
- python目录下无改动
- third_party/sunrise/backend，此部分不影响合入flagtree

#### 1.2 合入
对于主干代码的修改部分：
1. 修改的代码文件拷贝至`third_party/sunrise/backend/spec`目录下的对应位置
1. 对于所有有修改的主干代码的.cpp文件，都要在`spec/include`下对应的位置创建对应.h文件
    - .h文件内容为控制宏
1. 新增`spec/include/flagtree_spec.h`，内容为include所有上一步创建的.h文件
1. 把所有有修改的主干代码至原始状态，并用以下代码整体包裹

``` cpp
#if __has_include("flagtree_spec.h")
#include "flagtree_spec.h"
#endif

#ifndef FLAGTREE_SPEC_Triton_Dialect_TritonGPU_Transforms_Sunrise_Coalesce

// 原始代码 ...

#endif
```

对于闭源代码：

- 全部放在`third_party/sunrise/plugin`目录下
- 闭源代码 **【不能】** 放在github上
- 闭源代码单独产出一个`sunriseTritonPlugin.so`

### 二、Flagtree编译运行环境

#### 2.1 docker环境
宿主机需要安装ptpu docker环境。docker启动参数：

``` bash
docker run --rm -it --runtime=ptpu --env PTPU_VISIBLE_DEVICES=0 \
    -v /usr/local/tangrt:/usr/local/tangrt \
    -v /usr/local/pccl:/usr/local/pccl \
    -w /root  sunrise_triton:ubt_ptpu_v0.2 /bin/bash
```
可以通过-v选项把宿主机的代码映射到docker内，需要注意docker内的用户为root。

#### 2.2 构建安装flagtree
假设flagtree的项目目录在`/root/WorkSpace/flagtree`

``` bash
cd ~/WorkSpace/flagtree
source ~/WorkSpace/my/script/docker_triton_env.sh ~/WorkSpace/lb/commit_8929c/LLVM/build
FLAGTREE_PLUGIN=1 FLAGTREE_BACKEND=sunrise python3 setup.py develop
```

验证


### 三、提交pr流程