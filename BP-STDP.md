 # Abctract(�ʷ�)
SNN�� �ʱ� �ܰ迡 �ӹ��� �ִ� ���� �о��̸�, ��� Gradient descent�� ���� SNN���� �λ����� ������ ����������, 
������������ Ÿ������ �ʴٰ� �������� cost ��뵵 ũ��.  

���� �� �������� ��� ���(event-based) STDP ��Ģ�� IF neuron network�� �־ ���ο� suvervised learning approach�� ����.  

���ȵ� ��Ŀ��� temporal�ϰ� �������� learning rule�� �� time step���� ����Ǵ� backpropagation weight change update�� ����.   

�� ������� GD�� ������ STDP�� ������ ��� ì�� �� �ִ�.  

���� ����� XOR problem��, Iris data, �׸��� MNIST dataset�� ���ȵ� SNN�� ���� neuron network��ŭ ���������� �۵���.
# 1. Introduction
SNN�� ������������ Ÿ���ϰ�, ������ ȿ�������� ������ �������. �ٸ� SNN�� ������ �̻��� ������ũ�� ���� �۵��ϹǷ�,
��Ʈ��ũ �н��ϱ� ���ؼ� �������� ���� �̺� ������ Ȱ��ȭ �Լ��� �����ϰ� ���� �ʿ䰡 ����.  

multi-layer supervised learning�� SNN�� �ڱ��� ������ �����ϰ� �з��� �� �ִ� �ɷ��� �����ϱ� ���� �ʿ��� ����.  

SNN�� �ʱ� supervised learning�� �������� backpropagation�� ������ GD ������� �����Ͽ���. �߰������� ������ �� potential�� �Լ���
������ũ �ð��� ����ȭ�ؼ� ���� ��ǥ ������ũ�� ��� ������ũ ���� �Ÿ��� �ּ�ȭ��.  

GD�� �θ� ���Ǵ� supervised learning �̸� ��������, �¶��� ������� ������ ���̴� �� ������ ����.

�������� �н��� Victor & Purpura(VP) metric�� ����ϸ�, �¶��� �н��� Synapse current simulation�� ���� ������.

�¶��� GD�� multi-layer SNN���� supervised learning�� �����ϴ� �� �����Ͽ���.

������ memebrane potential�� ���Լ� ��� ���� ����� ������������ Ÿ������ ������ cost ����� ũ��.

�� �ٸ� ������δ� SNN�� ���� ������ Widrow-Hoff �н� ��Ģ�� Ȱ���Ͽ�, �����ĸ� ����ϰų�, �̻��� ������ũ�� �ش� �ƴҷα� ��ȣ�� ��ȯ�Ͽ�
���Ǿ�����, �̴� GD ��� ���ٹ����� ȿ������ ������ ��Ȯ���� ����.

�ٸ� ������δ� perceptron ��� ���ٹ��� �ִ�. �� ����� �� ������ũ �̺�Ʈ�� �ۼ�Ʈ�� �н��� ���� ���� �±׷� ����.

�� �𵨵��� single-layer supervised learning�� ����������, spike�� non-spike�� �з��ϴ� ������ STDP�� anti-STDP�� �����ϴ�
���ο� supervised learning���� Ȯ���.

STDP�� �ó��� �� ������ũ�� �ó��� �� ������ũ ���� �߻��ϸ� �ó����� ��ȭ�ϰ�, �׷��� ������ �ó����� ��ȭ��Ű�� �ۿ��� �Ͼ.
�� �������� IF neuron���� ������ SNN�� �н���Ű�� ���� STDP�� GD ������ ������ ���ο� multi-layer supervised learning�� ������.

����, IF neuron�� ReLU�� �ٻ����� ���̰�, backpropagation weight change rule���� ����� STDP/anti-STDP ��Ģ���� ���ǵǴ� �н� ���ٹ���
������ �� �ð� �ܰ迡�� ����� �� �ְ� ��.
# 2. Method
 spike-base�� �н� ��Ģ�� �����ϱ� ����, ������������ ��� IF neuron�� �� �˷��� ReLU activation function���� �ٻ��� �� �ִ�����
�������ڸ�, ó������ rate-based learning���� �ð��������� temporally local learning���� ��ȯ�ϴ� �ܰ踦 ����.  
## 2.1 Rectified Linear Unit versus IF neuron (ReLU vs IF neuron)
f(y) : neuron with ReLU activation function    
  
$x_h$ : input signal, $~w_h$ : $x_h$�� ���� ������ synaptic ����ġ�� ���������� ��Ÿ����,  
  
$$\begin{align}f(y) = max(0,~y),~y = \sum_{h} x_{h}w_{h}\end{align}$$   

$$\begin{align}\frac{\partial f}{\partial y} = \begin{cases} 1, & y > 0  \\ 0, & y \le 0 \end{cases}\end{align}$$

�̷������� IF neuron�� ReLU neuron�� �ٻ��� �� ����. Ư��, IF ������ membrane potential�� ReLU neuron�� Ȱ��ȭ ������ �ٻ� ����.

�̰��� �����ϱ� ���ؼ� LIF�� �ƴ� IF ������ $U(t)$��� membrane potential�� �Ӱ谪�� $\theta$ �� �ѱ�� spike train�� ������ ���� fire�ϰ� �ȴ�.  

$$\begin{align} &U(t) = U (t- \Delta t)~+~\sum_{h}w_{h}(t)s_{h}(t) \tag{3a}\\   
&if,~U(t)~\ge~\theta,~then,~r(t)~=~1,~U(t)~=~U_{rest} \tag{3b}  \end{align}$$  

���⼭ $s_h(t)$�� $r_h(t)$�� $t$ �ð��� ���� �ó��������� pre�� post neuron�� spike�� ���� �ǹ��Ѵ�. ���� $s_{h}(t),r_{h}(t) \in \{0,~1\}$ �̴�.
�߰������� �Ʒ� ÷�ڷ� ǥ�õǴ� h�� h��° �ó���? ����?�� �ǹ���.  

neuron�� membrane potential�� fire�� ��, ����(resting) potential�� $U_{rest}$���� �缳����. �� ��, $U_{rest}$�� 0���� ������.  

$G_{h}(t)$�� �ó������� pre-neuron�� spike train�� �ǹ��ϸ�, ���⼭ ���Ǵ� spike train�� �����ε� ��� ���Ǵ� ��������

������ũ �Ǵ� ������ ������ $t_{h}^p$�� ������ ���� �ִٰ� ���� �ȴ�. �׸��� ���⼭ ���Ǵ� $G_{h}(t)$�� ��-��Ÿ �Լ��� ������ ǥ���Ǵµ�,
�� ������ ������ �� ��, $\delta(t-t_{h}^{p})$�� ���ǹ��� ���� ������ ���ؼ���.  

��-��Ÿ �Լ��� $\delta$�� �Լ� �÷� ǥ���Ǵµ�, $\int_{-\infty}^{\infty} \delta(t)\,dt = 1 $,���⼭ $t$�� $t_{h}^{p}$�� ��ġ���� �ʴ´ٸ�,

$\delta(t-t_{h}^{p})$�� 0�� �ȴ�. �̸� ���� ������ũ �Ǵ� ������ ��Ƽ�, ������ũ �Ǵ� ������ ��� ���� �Լ��� ���� �� �ִ�. �� ������ �� (4)�̴�.

$$\begin{align} \\ G_h(t)~=~\sum_{t_{h}^{p}\in{s_{h}(t)=1}}\delta (t-t_{h}^{p}) \end{align}$$  

���⼭ $G_h(t)$�� spike train�̴�. ���� �м����ڸ�, $s_{h}(t) = 1$�� ������ ���� ��� pre-neuron�� ������ũ�� �ǹ��ϸ� $t$ ������ $h$��° ������ ������ũ�Ǿ��ٴ� ���� �ǹ��Ѵ�. 

���� $t_{h}^{p}$�� pre-neuron�� ������ũ �Ǵ� ������ ���ϰ�, ������ ���� ��-��Ÿ �Լ��� ����,
$G_h(t)$�� preneuron ������ũ�� �Ǵ� ������ �� �� �ִ�.

���� ���ذ� �Ǳ� ���� ���ø� �� �� �ִ�.  

���� ���, h��° ������ ������ 10ms���� 3ms�� 6ms �������� preneuron�� ������ũ �Ǿ��ٰ� ������ �ϸ�, $G_h(t)$�� $t=3ms, 6ms$�� ���� �����ϰ��
$G_h(t)$�� ���� 0�̴�. �׷��ٸ� $G_h(3)$�� ���� ��� �Ǵ°�? ��Ȯ���� ���� ���� �� �����Ƿ� 0�� �ƴϴٶ�� �����ϰ� �ѱ�� �ȴ�.
    
$$\begin{align}x_{h}~=~{1\over K}\int_{0}^{T} G_{h}(t^{\prime})\, dt^{\prime}\end{align}$$  

���⼭ $x_{h}$�� ������ũ Ƚ���� ��Ÿ����. �ٸ�, $K$�� ���� ����ȭ�� ������ ���� �ǹ��ϸ�, ��-��Ÿ�Լ��� ������ ǥ���� $G_{h}(t)$�� ��Ȯ�ϰ�
����ϱ� ���ؼ��� ������ �ؾ� �Ѵ�.   

�׷��� ó�� �ð��� $0ms$�� ������ �ð� $T ms$���� ������ �Ͽ�, �� ���̿� ������ũ�� �Ǿ�����, +1�� �Ǿ ��������� ������ �ð������� ������ũ Ƚ���� ��Ÿ����.  

���⼭ $K$�� �����ð� $Tms$ ���� �Ͼ �� �ִ� �ִ� ������ũ ���̴�. �ִ� ������ũ ���� ���ϱ� ���ؼ��� ������ũ�� ���� �ణ�� ���ذ� �ʿ��ѵ�, 

������ũ �ǰ� ������ ���� �� ��, ������ũ�� �ٽ� �Ͼ�� ���ؼ��� �ణ�� �ð��� �ʿ��ϴ�.  

���� �� �ð��� ����Ͽ� ���� �ִ� ������ũ ���̴�.

$$\begin{align}U(t) = \sum_{h}w_{h}(\int_{t-\alpha^{+}}^{t}\sum_{t_{h}^{p}}\delta(t^{\prime}-t_{h}^{p})\, dt^{\prime})\end{align}$$  

�� (6)������ post-neuron�� membrane potential�� ����� ���̴�. 

��Ȯ���� $t$������ post-neuron�� spike�� �Ͼ��, �� �ٷ� ������ ������ũ��
�Ͼ�� ������ $t-\alpha$��� �� �� �� ���̿� memebrane�� ������ $\theta$�� �ѱ��� �ʴ´�.

post-neuron�� membrane potential�� �ö󰡴� ������ pre-neuron�� potential�� threthold�� �Ѱܼ� ������ũ �ǰ� �׿� ���� ��������
post-neuron�� membrane potential�� �ö󰣴�. 

�׷��� ���� ����, pre-neuron�� ������ũ �Ǵ� ������, $t_{h}^{p}$�� ���ȴ�. 

���� ��꿡�� potential�� ���� ������ �Ͱ� ���� ����� �Ǹ�, �ܼ��ϰ� ������ �� (3a) ������
post-neuron�� ���ؼ��� �����Ų ������ $w_{h}$�� �� ����ġ�� �������� Ư�� ����ġ�� ���Ͽ� potential�� �����Ѵ�.

$$\begin{align}U^{tot} = \hat{y} = \sum_{t^{f}\in {r(t)=1}}U(t^{f})\end{align}$$

�� (7)�� post-neuron�� ������ũ �Ǵ� ���������� ������ ����� ���̴�. 

������ũ�� �Ͼ�ٴ� ���� potential�� threthold�� �Ѱ�ٴ� ���� �ǹ��ϸ�, �� (6)�� ���� spike�� �Ͼ�� ������ post-neuron�� potential�� ��� �� �� �ִ�.

����, $U^{tot}$�� post-neuron�� threthold�� �ѱ� potential�� ���� ������ ǥ���ȴ�. 

�׸��� threthold�� �ѱ� ���� potential�� ���� potential�� �Ǳ� ������, $U^{tot}$�� ������ ������ ������ ǥ���ȴ�. 

��, ���� ���, threthold $\theta$�� 10���� �����Ǿ� ���� ��, 10�� �ѱ�� ���� spike�� �Ͼ�� ������ 10�� ���� ���� ���̴�.

���� $U^{tot}$�� postsynaptic-neuron spike Ƚ���� $R$�� ������ ������ �� �� �ִ�.

$$\begin{align}f(\hat{y}) = \begin{cases} R = \gamma \hat{y} & \hat{y} > \theta \\ 0 & otherwise \end{cases}\end{align}$$

���� �� (8)�� ���� $U^{tot}$�� $R$�� ��� ���迡 �����ְ�, ���⼭ $\gamma$�� ��� ��� ���ҷν�, $T$�� ����ϰ� threthold�� $\theta$��
�ݺ�� �Ѵ�. ���� $\gamma \propto T \cdot \theta ^{-1}$ ��� ���� �����ȴ�.

���⼭ ��������� ǥ���� activation function�� ���� ReLU function���� $x$�� ��� �������� $\theta$��ŭ �̵��ϰ�,
���Ⱑ ���� ReLU�� $\gamma$ �� ��� ������ �����ϴٴ� ���� ������ �� ����.

Figure 1�� SNN�� activation function�� �װ��� ���Լ��� ����.

�� ������� ����, SNN�� activation function�� ReLU activation function�� �����ϴٴ� �� ���� �� �� �ְ�, 

�̸� ���� ReLU�� ����� ����� ���ο� ������ũ ��� learning rule�� �����Ͽ� ������ �� �ְ� ��.   

���� section������ �̸� ���� IF ������ ����Ǵ� STDP-base backpropagation rule�� ����.  
## 2.2 Backpropagation using STDP
 ���⼭�� ReLU Ȱ��ȭ�Լ��� ����ϴ� �Ű�������� backpropagation update rule���� ������ �޾�, STDP�� ����Ͽ���. 

 Figure 2���� �������� �Ű���� SNN�� ��Ʈ��ũ ������ �Ķ���͸� �� �� ����.  

 �� �� ��Ʈ��ũ�� �ֿ� �������� ������ ��� ��Ŀ� ����. �������� �Ű��(left)�� �Ǽ��� �Է� �� ������� ó���ϸ�, 
 SNN�� $T$ ms �ð� ���� ������ spike train�� �Է� �� ������� ó��.  
 
 GD�� ����ϴ� ANN�� ��ǥ�� $d$ �� ��°� $o$ ���� ���� ������ �ּ�ȭ�ϴ� ������ �ذ�.  
 
 $M$���� output neuron�� $N$���� training sample�� ���� ��, �Ϲ����� loss function�� ������ ����.  
 
 $$\begin{align}E = {1\over N} \sum_{k=1}^{N}\sum_{i=1}^{M}{(d_{k,i}-o_{k,i})}^2\end{align}$$

 �� �Ϲ����� ���Ŀ��� GD�� �н��� $H$���� �Է��� �����鼭 single training sample�� ���� ���� ��� neuron $i$�� ���, ����ġ ��ȭ ������ ������ ����.

$$\begin{align}E={(d_{i}-o_{i})}^2={(d_{i}-\sum_{h}o_{h}w_{ih})}^2\rarr\frac{\partial E}{\partial w_{ih}}=-2(d_{i}-o_{i})\cdot o_{h}\end{align}$$

 �̸� ���� $E~=~{(d_{i}-o_{i})}^2$��� ���� ����ġ update ������ �� �ִ�.

$$\begin{align}\Delta w_{ih}~\propto~-\frac{\partial E}{\partial w_{ih}}~\rarr~{\Delta}w_{ih}={\mu}(d_{i}-o_{i})o_{h} \end{align}$$

���⼭ �� ���� $d_{i}$, $o_{i}$, $o_{h}$�� ANN������ �������� Index�̴�. �̰��� spike train�� spike ���� $L_{i}$, $G_{i}$, $G_{h}$��
���� �ٲ� �� �ִ�. 

�̰��� [35]���� ���� ���̴�. 

�׷� ��� ������ ������ ����ġ ��ȭ ������ SNN���� synaptic weight update�� ����ϵ���
�籸���� �� �ִ�.

�� ������ Spiking IF neuron�� ReLU �������� �ٻ��ϴ� Theorem 1�� ���� ��ȿ�ϴ�.

��, �ٽ� ���� ������ 2.1���� ���� ������ ���� ANN�� weight change ������ SNN�� synaptic weight update �������� �ٻ��� �� ����.

SNN���� weight update rule�� ������ ����.

Post-synaptic neuron Spike Train : $$\begin{align} G_{i}(t) = \sum_{t_{i}^{p}{\in}\{r_{i}(t)=1\}}\delta(t-t_{i}^{p}) \tag{12a} \end{align}$$

Desired neuron Spike Train : $$\begin{align} L_{i}(t) = \sum_{t_{i}^{q}{\in}\{z_{i}(t)=1\}}\delta(t-t_{i}^{q}) \tag{12b} \end{align}$$

$$\begin{align}\\
{\Delta}w_{ih} = {\mu}\int_{0}^{T}(L_{i}(t^{\prime})\,dt^{\prime}{\cdot}\int_{0}^{T}G_{h}(t^{\prime})\,dt^{\prime}
\end{align}
$$

$$\begin{align}{\Delta}w_{ih}(t)~{\propto}~{\mu}(z_{i}(t)-r_{i}(t))s_{h}(t)\end{align}$$

$$\begin{align}{\Delta}w_{ih}(t) = {\mu}~{\cdot}~{\xi}_{i}(t) \sum_{t^{\prime}=t-{\epsilon}}^{t}s_{h}(t^{\prime})\end{align}$$

$$\begin{align}{\xi_{i}(t)=\begin{cases} 1, & z_{i}(t) = 1,~r_{i} \ne 1~in~[t-{\epsilon},t] \\ -1, & z_{i}(t) = 1,~r_{i} = 1~in~[t-{\epsilon},t] \\ 0, & otherwise \end{cases}}\end{align}$$

$$\begin{align}w_{ih}(t) = w_{ih}(t) + {\Delta}w_{ih}(t)\end{align}$$

$$\begin{align}{\Delta}w_{hj} = \mu~\cdot~(\sum_{i} \hat{\xi}_{i}w_{ih})~\cdot~o_{j}~\cdot~[o_{h} > 0] \end{align}$$

$$\begin{align}{\Delta}w_{hj} = \mu \int_{0}^{T}(\sum_{i}\xi_{i}(t^{\prime})w_{ih}(t^{\prime}))\, dt^{\prime} \cdot \int_{0}^{T}(\sum_{t_{j}^{p}}\delta(t^{\prime}-t_{j}^{p}))\, dt^{\prime} \cdot ([\int_{0}^{T}\sum_{t_{h}^{p}}\delta(t^{\prime}-t_{h}^{p})\, dt^{\prime}] > 0)\end{align}$$

$$\begin{align}{\Delta}w_{hj}(t) = \begin{cases} {\mu} \cdot \sum_{i}\xi_{i}(t)w_{ih}(t) \cdot \sum_{t^{\prime}=t-{\epsilon}}^{t}s_{j}(t^{\prime}), & s_{h} = 1~in~[t-{\epsilon},t] \\ 0, & otherwise \end{cases}\end{align}$$

$$\begin{align}w_{hj}(t) = w_{hj}(t) + {\Delta}w_{hj}(t) \end{align}$$
