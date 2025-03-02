# pybind
smash together python3 + numpy + C++ + Eigen + ultimately CUDA

## environment
https://lightning.ai/jiahangl/vision-model/studios/civilian-orange-d1xo/code
```
cat /etc/lsb-release

DISTRIB_ID=Ubuntu
DISTRIB_RELEASE=20.04
DISTRIB_CODENAME=focal
DISTRIB_DESCRIPTION="Ubuntu 20.04.6 LTS"
```

## install bazel
```
sudo apt install apt-transport-https curl gnupg -y
curl -fsSL https://bazel.build/bazel-release.pub.gpg | gpg --dearmor >bazel-archive-keyring.gpg
sudo mv bazel-archive-keyring.gpg /usr/share/keyrings
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/bazel-archive-keyring.gpg] https://storage.googleapis.com/bazel-apt stable jdk1.8" | sudo tee /etc/apt/sources.list.d/bazel.list
sudo apt update && sudo apt install bazel
```

## build and run
```
bazel run main

INFO: Running command line: bazel-bin/main
mp.add(2, 3)=5
mp.matmul_generic(x, y)=array([[-2.],
       [-2.],
       [-2.]])
mp.matmul_cuda(x, y)=array([[-2.],
       [-2.],
       [-2.]])
x @ y =array([[-2.],
       [-2.],
       [-2.]])
mp.vector_add_cuda(y, y)=array([ 2.,  0., -2.])
y + y =array([[ 2.],
       [ 0.],
       [-2.]])
```

```
bazel run benchmark

INFO: Running command line: bazel-bin/benchmark
numpy elapsed: 0.02 sec
mumpy.matmul_generic elapsed: 15.06 sec
(z_mp_cpu_gen - z_np).mean()=np.float64(-2.0200859600951216e-17)
mumpy.matmul_row elapsed: 14.72 sec
(z_mp_cpu_row - z_np).mean()=np.float64(-2.0200859600951216e-17)
mumpy.matmul_col elapsed: 13.57 sec
(z_mp_cpu_col - z_np).mean()=np.float64(-2.0200859600951216e-17)
mumpy.matmul_cuda elapsed: 0.14 sec
(z_mp_gpu - z_np).mean()=np.float64(1.9976171135296154e-17)
```
