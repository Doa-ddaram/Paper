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

$$\begin{align}\frac{\partial f}{\partial y} = \left \lbrace \begin{align*} &1,~y > 0  \\ &0,~y \le 0 \end{align*} \right. \end{align}$$

�̷������� IF neuron�� ReLU neuron�� �ٻ��� �� ����. Ư��, IF ������ membrane potential�� ReLU neuron�� Ȱ��ȭ ������ �ٻ� ����.

�̰��� �����ϱ� ���ؼ� LIF�� �ƴ� IF ������ $U(t)$��� membrane potential�� �Ӱ谪�� $\theta$ �� �ѱ�� spike train�� ������ ���� fire�ϰ� �ȴ�.  

$$\begin{align} & \begin{equation*} U(t) = U (t- \Delta t)~+~\sum_{h}w_{h}(t)s_{h}(t) \end{equation*} \tag{3a} \\&
\begin{equation*}if~U(t)~{\ge}~{\theta}~then,~r(t)~=~1,~U(t)~=~U_{rest}\end{equation*} \tag{3b}  \end{align}$$  

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

$$\begin{align}E={(d_{i}-o_{i})}^2={(d_{i}-\sum_{h}o_{h}w_{ih})}^2\rightarrow\frac{\partial E}{\partial w_{ih}}=-2(d_{i}-o_{i})\cdot o_{h}\end{align}$$

 �̸� ���� $E~=~{(d_{i}-o_{i})}^2$��� ���� ����ġ update ������ �� �ִ�.

$$\begin{align}\Delta w_{ih}~\propto~-\frac{\partial E}{\partial w_{ih}}~\rightarrow~{\Delta}w_{ih}={\mu}(d_{i}-o_{i})o_{h} \end{align}$$

���⼭ �� ���� $d_{i}$, $o_{i}$, $o_{h}$�� ANN������ �������� Index�̴�. �̰��� spike train�� spike ���� $L_{i}$, $G_{i}$, $G_{h}$��
���� �ٲ� �� �ִ�. 

�̰��� [35]���� ���� ���̴�. 

�׷� ��� ������ ������ ����ġ ��ȭ ������ SNN���� synaptic weight update�� ����ϵ���
�籸���� �� �ִ�.

�� ������ Spiking IF neuron�� ReLU �������� �ٻ��ϴ� Theorem 1�� ���� ��ȿ�ϴ�.

��, �ٽ� ���� ������ 2.1���� ���� ������ ���� ANN�� weight change ������ SNN�� synaptic weight update �������� �ٻ��� �� ����.

SNN���� weight update rule�� ������ ����.

Post-synaptic neuron Spike Train : $$\begin{align}G_{i}(t)=\sum_{t_{i}^{p}{\in}\{r_{i}(t)=1\}}\delta(t-t_{i}^{p})\tag{12a}\end{align}$$

Desired neuron Spike Train : $$\begin{align} L_{i}(t) = \sum_{t_{i}^{q}{\in}\{z_{i}(t)=1\}}\delta(t-t_{i}^{q}) \tag{12b} \end{align}$$

�׷� ���� ������� weight update�� 0���� ���� �ð� $T$������ ������ ���� weight update���� ����ȭ�� �� ����.

$$\begin{align}\\
{\Delta}w_{ih} = {\mu}\int_{0}^{T}(L_{i}(t^{\prime})\,dt^{\prime}{\cdot}\int_{0}^{T}G_{h}(t^{\prime})\,dt^{\prime}
\end{align}
$$

������ �� (13)�� �ð����� ���鿡�� local���� �ʰ� ���� �ð� $T$ ���θ� �ٷ�. 

���� �̰��� localizing �ϱ� ����, �ð� ���� $T$�� ����ȭ��.

����ȭ�ϴ� ����� $T$�� �� �ð� ���ݿ� �ϳ� �Ǵ� 0���� ������ũ�� �����ϵ��� ������.

�̷��� �ϴ� ������ ������ ������ũ�� �Ͼ�� 1, ������ũ�� �Ͼ�� ������ 0���� �Ǳ� ������, Spike �ϴ� �Ϳ� Ÿ�缺�� ���� �̷��� ����ȭ��.

���� �� (13)�� ����ȭ�� learning rule�� �� (14)���� ��üȭ��.


$$\begin{align}{\Delta}w_{ih}(t)~{\propto}~{\mu}(z_{i}(t)-r_{i}(t))s_{h}(t)\end{align}$$

�� ������ �����ϱ� ���� event-based STDP�� anti-STDP�� combination�ߴ�.

�� ����� teacher signal�� �����, STDP�� anti-STDP ���̸� ��ȯ�ϸ� synapse weight�� update��.

��, ��ǥ�� �Ǵ� neuron�� STDP�� ������, ��ǥ�� �ƴ� neuron�� anti-STDP�� ����.

�츮�� desire�� spike train $z$�� �Է� �󺧿� ���� ���ǰ� �Ǵµ�, ��ǥ neuron�� �ִ� ������ũ �󵵸� ������ ($\beta$)�� ���� ��Ÿ����, ���ǥ Ÿ���� silent��.

�߰������� learning rule�� desired spike time $z_{i}(t)$���� �۵���. ��, ��� ��ǥ �������� desire�ϴ� spike time�� ������ time��.

�� (15)�� supervised SNN�� ��� layer�� ������ weight update ���̴�.

$$\begin{align}{\Delta}w_{ih}(t) = {\mu}~{\cdot}~{\xi}_{i}(t) \sum_{t^{\prime}=t-{\epsilon}}^{t}s_{h}(t^{\prime})\end{align}$$

���⼭ ���̴� $\xi_{i}(t)$�� �� (16)���� ���ǵȴ�. 

$$\begin{align}{\xi_{i}(t)=\begin{cases} 1, & z_{i}(t) = 1,~r_{i} \ne 1~in~[t-{\epsilon},t] \\ -1, & z_{i}(t) = 0,~r_{i} = 1~in~[t-{\epsilon},t] \\ 0, & otherwise \end{cases}}\end{align}$$

�� (16)�� �ؼ����ڸ�, ��ǥ�� �ƴ� ������ ���� $T$���� ������ũ �ǰ�,���� pre-neuron�� ���� $T$���� ������ũ �Ǵ°��, -1�� ���ǵ�.  
���� ��ǥ ������ ���� $T$���� ������ũ �ǰ�, pre-neuron�� ���� $T$���� ������ũ�� ���� ���� ���, 1�� ���ǵ�.

���� ���������� ��� layer�� synaptic weight�� �� (17)�� ���� �̷�����.

$$\begin{align}w_{ih}(t) = w_{ih}(t) + {\Delta}w_{ih}(t)\end{align}$$

��ǥ neuron�� $z_{i}(t)�� ���� �����Ǵµ�, $z_{i}(t)=1$�� ��ǥ neuron�� �ǹ��ϰ� $z_{i}(t)=0$�� ���ǥ neuron�� �ǹ���.

��� layer�� weight change�� desired spike time���� �����.

Desired spike time $t$���� ��ǥ neuron�� STDP window�� �˷��� �ſ� ª�� �ð� ���� $[t-\Delta,t]$ �ȿ� �̷���� �Ѵ�.

�׷��� ���� ���, synaptic weight�� STDP window���� pre-synaptic neuron Ȱ��(�ַ� ������ũ�� �Ͼ�� �� �Ͼ�Ŀ� ����)�� ����� ������.

presynaptic Ȱ���� STDP window���� presynaptic�� spike Ƚ���� ��Ÿ���� $\sum_{t^{\prime}=t-\epsilon}^{t}(s_{h}(t^{\prime}))$���� ǥ����.

�ݸ鿡, ���ǥ neuron�� ��ȭ�Ǹ� ������ ������� weight depression�� �Ͼ. 

�� ����� �������� Gradient descent���� �����ϸ�, SNN���� �ð������̰� local learning�� �����Ѵ�.

�� (18)������ ReLU neuron�� �������� ����Ǵ� backpropagation weight update rule�̴�. 

$$\begin{align}{\Delta}w_{hj} = \mu~\cdot~(\sum_{i} \hat{\xi}_{i}w_{ih})~\cdot~o_{j}~\cdot~[o_{h} > 0] \end{align}$$

$\hat{\xi_{i}}$�� desired�� ��� value ���� ������ $(d_{i}-o_{i})$�� ��Ÿ����.

�׸��� �츮�� SNN, $\hat{\xi_{i}}$�� �� (16)�� ���� \xi_{i}�� �ٻ�ȴ�. 

value $[o_{h} > 0]$�� �������� ReLU ������ ���Լ��� ǥ���ȴ�. 

IF neuron�� ReLU reuron���� �ٻ��ϸ� (�� (8)�� ����), ���� SNN�� ������ũ Ƚ���� �������� 

$$\begin{align}{\Delta}w_{hj} = \mu \int_{0}^{T}(\sum_{i}\xi_{i}(t^{\prime})w_{ih}(t^{\prime}))\, dt^{\prime} \cdot \int_{0}^{T}(\sum_{t_{j}^{p}}\delta(t^{\prime}-t_{j}^{p}))\, dt^{\prime} \cdot ([\int_{0}^{T}\sum_{t_{h}^{p}}\delta(t^{\prime}-t_{h}^{p})\, dt^{\prime}] > 0)\end{align}$$

�ð� $T$�� ���� ���� $[t-\epsilon,~t]$���� ���� ��, �������� synaptic weight�� �ð� ���鿡�� local rule�� ���� Update��.

$$\begin{align}{\Delta}w_{hj}(t) =  \left \lbrace \begin{align*}&{\mu} \cdot \sum_{i}\xi_{i}(t)w_{ih}(t) \cdot \sum_{t^{\prime}=t-{\epsilon}}^{t}s_{j}(t^{\prime}),~s_{h} = 1~in~[t-{\epsilon},t] \\ &0,~otherwise \end{align*} \right. \end{align}$$

����������, �������� synaptic weight�� �� (21)�� ���� update��.

$$\begin{align}w_{hj}(t) = w_{hj}(t) + {\Delta}w_{hj}(t) \end{align}$$

���� learning rule�� �������� neuron h�� ��ȭ�ϸ� (postsynaptic neuron���� spike�� �߻��ϸ�) 0�� ���� �ʴ´�.

����, weight�� presynaptic($s_{j}(t)$)�� postsynaptic($s_{h}(j)$) spike time�� ���� ������Ʈ��. �׸��� �̰��� ǥ�� STDP ��Ģ�� ����.

�߰������� ReLU�� ���Լ��� $o_{h} > 0 $�� �� (20)�� ���ǰ� ���� IF neuron���� spike generation�� ������.

�� weight change rule�� ���� �츮�� BP-STDP��� �Ҹ��� STDP�� ������� Backpropagation algorithm�� ����� ���� SNN�� �������.

Figure 3�� �ϳ��� �������� ���� SNN�� ������ BP-STDP�� ��Ÿ��.

# 3. Result
������ ���� XOR ������, iris dataset, MNIST dataset �̷��� 3���� �ٸ� �����鿡 ���ؼ� ������ �����Ͽ���.

���迡�� ���� parameter�� Table 1�� ���� �̷�����. 

��� ���迡�� synaptic weight �ʱ�ȭ�� ��� $0$�̰� ǥ�������� $1$�� Gaussian function���� ���������� ���ڸ� ��������.

## 3.1 $XOR~problem$
BP-STDP �˰����� XOR problem�� ���� ���������� �и����� �ʴ� ������ �ذ��� �� �ִ� �ɷ��� �����ٴ� ���� �����ֱ� ���� �ǽõǾ���.

dataset�� {(0.2, 0.2), (0.2, 1), (1, 0.2), (1, 1)}�� 4������ datapoint�� �����ϸ�, �� point�� �°� {0, 1, 1, 0}���� ���ε�.

�츮�� IF neuron Ȱ��ȭ�� ���� (��, spike �߻��� ����) datapoint�� �Ǽ����� 0.2�� 0���� �����Ͽ���.

�Ű�� ������ 2���� �Է�, 20���� ������ ����, 2���� ��� IF neuron���� �����Ͽ���.

�� �������� �������� neuron�� ���� ����� ū ������ ��ġ�� ����.

��� �������� MNIST �з� task���� hidden reuron �� ������ ������.

�� �Է� neuron���� spike train�� �׿� ������ �Է� ���� ���� ��ȭ��.

���� ��� value 1�� ���, ������ũ �Ͼ�ٴ� �ǹ̷� ���� �ִ� ������ũ ����(250Hz)�� �����ٰ� ǥ����.

Figure 4�� �н������� �� �� �ִµ�, �� ���ڴ� 4������ �Է� spike pattern�� ���� �� ���� ��� neuron Ȱ���� ��Ÿ��.

�� 150 ���� �н� iteration ����, ��� ������ �Է¿� ���� ���������� �����ϰ� ��.

Figure 5�� �� (22)�� ���ǵ� ������ �Լ��� ����ϸ鼭 learning ���� ������ ��Ÿ��.

�� Figure�� ����, ������ learning rate($\mu$)�� $[0.01, 0.0005]$�� ����.

$$\begin{align}MSE = {1 \over N} \sum_{k=1}^{N}{({1 \over T} \sum_{t = 1}^{T}\xi^{k}(t))}^{2} ,\xi^{k}(t)=\sum_{i} \xi_{i}^{k}(t)
\end{align}$$

�� (22)����, $N$�� $\xi_{i}^{k}(t)$�� ���� �н� batch size�� $k$���� sample�� ���� ��� neuron $i$�� ���� ���̴�.

## 3.2 $Iris~dataset$
 Iris dataset�� (Setosa, Versicolour, Virginica)��� 3������ �ٸ� �� ������ ������ ������, petal�� sepal�� ���̿� ���̸� ��Ÿ��. (�� 4���� Ư¡�� ����).

 �� Ư¡ ���� 0�� 1 ���̷� ����ȭ�س���, �Է� ������ũ train�� ������.
 
 �� ���迡�� �Ű�� ������ 4���� �Է�, 30���� ������, 3���� ��� IF neuron���� ������.

�ʱ� ���迡���� 10�� �̻��� ������ ������ ����ϴ� ���� ��Ȯ���� ũ�� ����Ű�� �ʾ���.

XOR ������ ��� �����ϰ� Figure 6�� �н��� ���� SNN�� �� ���� �� ���Ͽ� ���� �����Ͽ� �����ϰ� ��.

Figure 7�� �� (22)�� ���ǵ� MSE�� �������� SNN�� �н� ������ ��Ÿ��. 

���������� 5���� fold cross validation ����� 96% ��Ȯ���� �����ָ�, �������� Backpropagation�� ����ϴ� �Ű���� 96.7%�� ��Ȯ���� ������.

�� ��� BP-STDP �˰����� �ð��� SNN�� ���������� �н���ų �� ������ ������.

�Դٰ� Table 2���� BP-STDP�� �ٸ� spike-based�� �������� supervised learning�� ���Ͽ���.

BP-STDP�� �ٸ� �ð����� GD-based approach���� cost ȿ�������� ���� ������, ��κ��� ���� ������� �� ���ų� ������ ������ ������.

## 3.3 $MNIST Dataset$

�� �����麸�� �� ������ �������� �˰����� �����ϰ� �ذ�Ǵ� �� ���ϱ� ���� MNIST dataset�� ���� SNN ������ 784���� �Է� neuron��,
100���� 1500������ ������ neruon�� 10���� ��� IF neuron���� ������ BP-STDP�̴�.

SNN�� 6�� ���� training sample���� training�ϰ�, �� ���� testing sample���� testing sample���� testing����.

�Է� spike train�� ����ȭ�� pixel �� ($[0,1]$ ����)�� ����Ͽ� ������ ������ũ������ ���� ����(random lags)�� ����� ��������.

Figure 8�� ���������� ���õ� digit�κ��� ������ spike train�� ���� �н� �� ��� neuron�� membrane potential�� ������.

��ǥ neuron�� membrane potential�� ������ ����Ͽ� threthold�� �����ϰ�, �� ���� �ٸ� �������� Ȱ��(potential)�� ���� 0�� ������.

�� ���� ���� ($< 9 ms$)�� �Ű���� ���� ������ ����.

Figure 9a�� 1200 �н� epoch�� �н��� ������ �� �� ����. �� epoch�� 50���� MNIST digit�� ��Ÿ��.

�� �׷������� MSE ��Ʈ�� �н����� 0.001�� 0.0005�� ��, ���� ������ �����ϴ� ���� �� �� ����.

Figure 9b�� �н����� 1000���� ������ neuron�� ���� SNN���� ��Ȯ���� MSE ���� �� �� ����.

100�� 900 epoch�� ���� ��, ���� ������ 90%�� 96% �޼��� ���� �� �� ����.

�������� neuron�� ���� ������ �����ϱ� ���� 100���� 1500���� ������ IF neuron�� ������ 6���� SNN�� BP-STDP�� �����Ͽ���.

Figure 10�� �� SNN���� �н��� ���� ��Ȯ���� �� �� ����. �ְ� ��Ȯ���� 500�� �̻��� ������ neuron�� ������ �Ű���̾���.

���������� BP-STDP �˰����� 2���� layer �Ǵ� 3���� layer�� ������ SNN���� �򰡵Ǿ���.

SNN ����(��Ȯ���� ������ neuron�� ��)�� �� ���� �񱳸� �����ϱ� ���� [39]���� ���� �Ű�� ������ �����ϰ� �Ͽ���.

BP-STDP�� 2���� layer SNN������ $96.6 \pm 0.1%$�� �޼��Ͽ�����, 3���� layer SNN������ $97.2 \pm 0.07 %$�� �޼��Ͽ���.

�� ����� backpropagation���� �н��� �������� �Ű���� ������ ��Ȯ���� ��Ÿ��.

Table 3�� ���ȵ� supervised learning(BP-STDP)�� �������� backpropagation �Ű��(�Ǵ� GD), Deep SNN�� ����� spiking backpropagation �����
�ֱ� MNIST �з��� ���� STDP-based SNN�� ���Ͽ���.

�� �񱳴� �ð��� SNN ������ �����ϰ� ������������ �������� BP-STDP�� ������ ������.

BP-STDP�� ���� SNN�� ������ end-to-end STDP based, supervised learing approach��.

[40,41]���� �Ұ��� SNN�� Ư¡ ������ ���� ���� STDP �н��� ������������, �������� supervised layer���� SVM �з��⸦ ����Ͽ���.

��� GD approach�� ������������, ���� ȿ������ STDP learning�� �ٸ��� �������� ������ �������� ����. 

Ư�� �ֱٿ��� backpropagation�� ����ϴ� Deep SNN�� BP-STDP�� ȿ������ �� ���� ȿ�Ｚ�� ���� ��

������ �̰��� neuron�� ������ũ event�� ����ϴ� �� ��ſ� �׵��� membrane potential�� ����Ͽ� Ȱ��ȭ �Լ��� ���Լ��� �����.

# 4. Discussion

BP-STDP�� SNN�� novel�� supervised learning���� �Ұ���.

�װ��� ��÷�� �������� GD approach�� ���Ҹ� ���� ������ ������.

BP-STDP�� �������� ������ �����ϴ� local learning�� �ϴµ�, local learning�� IF neuron�� ���� layer���� spike rate �Ӹ� �ƴ϶� spike time���� ����ϴ� �����.

Bengio et al.���� synaptic weight update�� presynaptic spike event�� postsynaptic �ð��� Ȱ���� ����Ѵٴ� ���� ������.

�׸��� �̰��� STDP ��Ģ�� �����ϸ�, STDP�� postsynaptic �ð��� spike rate�� �����Ǿ� �ִٴ� Hioton�� ������ Ȯ����.

���ȵ� �˰����� ReLU neuron�� ���Ե� �������� �Ű������ ���� backpropagation rule���� ������ �޾���.

������ �̰��� ������������ ���Ҽ��ϰ� �ð������� local learning rule�� ������.

�̷��� ������ SNN���� spike based ������ �����ϱ� ���� IF neuron�� ReLU neuron���� �ٻ��Ͽ� �ʱ⿡ �̷�����.

spiking supervised learning�� �ð������� presynaptic�� postsynaptic neuron spike event�� �����Ǵ� spiking neural layer�� ����Ǵ� STDP�� anti-STDP�� ȥ���Ͽ� ������.

����, �� ���������� �ð������� local�� STDP�� ȿ������ GD�� ��Ȯ�� ��ο� ���� ������ ����.

�ֵ� ������, ���� �Ű�� �������� IF neuron�� spiking event�� ���� ���� propagation�� ��� �����Ǿ��°�?�̴�.

�� ������ ���ϱ� ���� ���� ���� ������ ��ȭ�ϰų�$(\sum_{i}\xi_{i} > 0)$ �����ϴ� ��ȣ$(\sum_{i}\xi_{i} <> 0)$�� �����غ���.

�� �� ������ �ڱ�(����)�ϴ� ���� synapse�� input weight�� presynaptic spike event�� ���� ����Ͽ� �����Ǵ� membrane potential�� ����(����)��Ű�� ���� �ǹ���.

���� BP-STDP update rule�� synaptic weight�� �ð������� ���� ��ȣ�� presynaptic spike time�� ���� ��ȭ�ϸ�, �� �ð� �ܰ��� �������� �ൿ potential�� ������.

���� ����� ���� SNN�� supervised learning�� �����Կ� �־� BP-STDP�� ������ ������.

XOR problem�� BP-STDP�� spike train���� ǥ���� ���������� �и����� �ʴ� sample�� �ð� ���� $T$ ms �ð� ���� ���� �з��� �� �ִ� �ɷ��� ��������.

IRIS�� MNIST �з��� ���� ������ ������ �������� backpropagation �˰���� �ֱ� SNN�� ���Ҹ��� ������ ������ ���� �����ָ�, BP-STDP�� spiking pattern classification�� ���� end-toend STDP-based supervised learning�� ������.

������������ ������ �ް� �ð������� local�� STDP rule�� ������ �߻��ϴ� ȿ������ ��꿡 �� ���� �� ��������� ��.

�츮�� ���� �ȿ�����, �� ���ٹ��� cost������ ����� ���� GD�� ���ϸ鼭 ���� STDP-based supervised learning�� �����ϴ� ������ �����.

# 5. Conclusion

�� ���� ���� ������ �ൿ�� spike rate�� ���εǸ� IF neuron�� ReLU�� �ٻ��� �� ������ ����.

���� spiking IF neuron�� �Ű���� �������� �Ű���� ����� backpropagation learning�� ����ϴٰ� �� �� ����.

IF neuron�� ���� ���� SNN���� ����� STDp�� antiSTDP�� �������� �ð����̰� local�� learning rule�� ��������.

BP-STDP�� spiking neuron���� ȿ������ STDP rule�� ����Ͽ� �������� ������ ������ ���, ���� �Ű���� �н��ϸ鼭 GD�� ���� �����.

���� GD-based weight change rule�� spike-based STDP rule���� ��ȯ�� spiking GD rule�� �������� �� ����, cost������ ����� �����.

XOR ���������� ������ ���ȵ� SNN�� ���������� �ʰ� �и��� pattern�� �з��� �� ������ ������.

�׸��� Iris�� MNIST dataset������ ���� �򰡴� ��÷���� ����� spiking neuron�� ���� �Ű���� ���� �� �ִ� ���� �з� ��Ȯ���� ������.

BP-STDP ���� ����� BP-STDP�� ����ȭ module�� �߰��� ���� SNN�� �����ϱ� ���� ���� ������ �ʿ伺�� ������.

���� SNN�� �� ū pattern recongnition �۾��� Ȱ���� �� ������, ���� ������ ȿ������ ����� ������ �� ����.