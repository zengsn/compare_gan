# Compare GAN by Google

The new tips and workaround to reproduce the training from scratch, mainly in mainland of China. More info can be found in [README-Google.md](./README-Google.md). 

## 1. Goals

- Make it work in mainland, China. 
- Fix some issues locally. 
- Discuss with ones who in the same environment. 
- Contribute to cutting-edge GAN research. 

## 2. Tips

### 2.1 Pick the right versions

Install the prerequired libraries via 
```  
pip install -e .
```  

Make sure you are using the following versions:  
```   
sudo apt install cuda-10-0  
pip install tensorflow-gpu==1.13.1  
```   

Install newer version of tensorflow-datasets:
```   
pip install tensorflow-datasets==1.0.2
```   

## 2. Known Issues

- See the [commit](https://github.com/zengsn/compare_gan/commit/fb34717b4863312c681f8174dfbb1984d9dddeed) to fix 
```  
TypeError: '<=' not supported between instances of 'int' and 'str'
```  

