 # Abctract(초록)
SNN은 초기 단계에 머물러 있는 연구 분야이며, 비록 Gradient descent가 심층 SNN에서 인상적인 성능을 보여줬지만, 
생물학적으로 타당하지 않다고 여겨지며 cost 비용도 크다.  

따라서 이 논문에서는 사건 기반(event-based) STDP 규칙을 IF neuron network에 넣어서 새로운 suvervised learning approach를 제안.  

제안된 방식에서 temporal하고 지역적인 learning rule은 매 time step마다 적용되는 backpropagation weight change update를 따름.   

이 방식으로 GD의 이점과 STDP의 이점을 모두 챙길 수 있다.  

실험 결과는 XOR problem과, Iris data, 그리고 MNIST dataset에 제안된 SNN이 기존 neuron network만큼 성공적으로 작동함.
# 1. Introduction
SNN은 생물학적으로 타당하고, 에너지 효율성으로 관심을 끌어왔음. 다만 SNN의 뉴런은 이산적 스파이크를 통해 작동하므로,
네트워크 학습하기 위해선 연속적인 값의 미분 가능한 활성화 함수로 구성하게 만들 필요가 있음.  

multi-layer supervised learning은 SNN이 자극의 패턴을 예측하고 분류할 수 있는 능력을 입증하기 위해 필요한 개념.  

SNN의 초기 supervised learning은 전통적인 backpropagation과 유사한 GD 방법으로 개발하였음. 추가적으로 뉴런의 막 potential을 함수로
스파이크 시간을 공식화해서 단일 목표 스파이크와 출력 스파이크 사이 거리를 최소화함.  

GD는 널리 사용되는 supervised learning 이며 오프라인, 온라인 방식으로 성능을 높이는 데 여전히 사용됨.

오프라인 학습은 Victor & Purpura(VP) metric을 사용하며, 온라인 학습은 Synapse current simulation을 통해 수행함.

온라인 GD는 multi-layer SNN에서 supervised learning을 개발하는 데 성공하였다.

하지만 memebrane potential과 도함수 기반 접근 방식이 생물학적으로 타당하지 않으며 cost 비용이 크다.

또 다른 방식으로는 SNN에 대해 수정된 Widrow-Hoff 학습 규칙을 활용하여, 역전파를 사용하거나, 이산적 스파이크를 해당 아닐로그 신호로 변환하여
사용되었으며, 이는 GD 기반 접근법보다 효율성은 높지만 정확도가 낮음.

다른 방식으로는 perceptron 기반 접근법이 있다. 이 방식은 각 스파이크 이벤트를 퍼셉트론 학습을 위한 이진 태그로 본다.

이 모델들은 single-layer supervised learning을 제공하지만, spike와 non-spike를 분류하는 개념은 STDP와 anti-STDP를 통합하는
새로운 supervised learning으로 확장됨.

STDP는 시냅스 전 스파이크가 시냅스 후 스파이크 전에 발생하면 시냅스를 강화하고, 그렇지 않으면 시냅스를 약화시키는 작용이 일어남.
이 논문에서는 IF neuron으로 구성된 SNN을 학습시키기 위해 STDP와 GD 장점을 결합한 새로운 multi-layer supervised learning을 제안함.

먼저, IF neuron이 ReLU를 근사함을 보이고, backpropagation weight change rule에서 도출된 STDP/anti-STDP 규칙으로 정의되는 학습 접근법을
개발해 각 시간 단계에서 적용될 수 있게 함.
# 2. Method
 spike-base의 학습 규칙을 제안하기 전에, 생물학적으로 어떻게 IF neuron이 잘 알려진 ReLU activation function으로 근사할 수 있는지를
보여주자면, 처음으로 rate-based learning에서 시공간적으로 temporally local learning으로 변환하는 단계를 가짐.  
## 2.1 Rectified Linear Unit versus IF neuron (ReLU vs IF neuron)
f(y) : neuron with ReLU activation function    
  
$x_h$ : input signal, $~w_h$ : $x_h$를 통해 연관된 synaptic 가중치를 수식적으로 나타내면,  
  
$$\begin{align}f(y) = max(0,~y),~y = \sum_{h} x_{h}w_{h}\end{align}$$   

$$\begin{align}\frac{\partial f}{\partial y} = \left \lbrace \begin{align*} &1,~y > 0  \\ &0,~y \le 0 \end{align*} \right. \end{align}$$

이론적으로 IF neuron은 ReLU neuron을 근사할 수 있음. 특히, IF 뉴런의 membrane potential은 ReLU neuron의 활성화 값으로 근사 가능.

이것을 증명하기 위해서 LIF가 아닌 IF 뉴런은 $U(t)$라는 membrane potential이 임계값인 $\theta$ 를 넘기면 spike train에 흔적이 남고 fire하게 된다.  

$$\begin{align} & \begin{equation*} U(t) = U (t- \Delta t)~+~\sum_{h}w_{h}(t)s_{h}(t) \end{equation*} \tag{3a} \\&
\begin{equation*}if~U(t)~{\ge}~{\theta}~then,~r(t)~=~1,~U(t)~=~U_{rest}\end{equation*} \tag{3b}  \end{align}$$  

여기서 $s_h(t)$와 $r_h(t)$는 $t$ 시간에 대해 시냅스적으로 pre와 post neuron의 spike를 각각 의미한다. 따라서 $s_{h}(t),r_{h}(t) \in \{0,~1\}$ 이다.
추가적으로 아래 첨자로 표시되는 h는 h번째 시냅스? 뉴런?을 의미함.  

neuron의 membrane potential은 fire할 때, 휴지(resting) potential인 $U_{rest}$으로 재설정됨. 이 때, $U_{rest}$은 0으로 생각함.  

$G_{h}(t)$은 시냅스에서 pre-neuron의 spike train을 의미하며, 여기서 사용되는 spike train은 앞으로도 계속 사용되는 개념으로

스파이크 되는 지점의 시점인 $t_{h}^p$의 정보를 갖고 있다고 보면 된다. 그리고 여기서 사용되는 $G_{h}(t)$는 디랙-델타 함수의 합으로 표현되는데,
그 이유는 적분을 할 때, $\delta(t-t_{h}^{p})$를 유의미한 값을 가지기 위해서다.  

디랙-델타 함수는 $\delta$의 함수 꼴로 표현되는데, $\int_{-\infty}^{\infty} \delta(t)\,dt = 1 $,여기서 $t$가 $t_{h}^{p}$와 일치하지 않는다면,

$\delta(t-t_{h}^{p})$가 0이 된다. 이를 통해 스파이크 되는 지점을 잡아서, 스파이크 되는 시점을 모아 놓은 함수를 얻을 수 있다. 그 내용이 식 (4)이다.

$$\begin{align} \\ G_h(t)~=~\sum_{t_{h}^{p}\in{s_{h}(t)=1}}\delta (t-t_{h}^{p}) \end{align}$$  

여기서 $G_h(t)$는 spike train이다. 식을 분석하자면, $s_{h}(t) = 1$는 위에서 말한 대로 pre-neuron의 스파이크를 의미하며 $t$ 시점의 $h$번째 뉴런이 스파이크되었다는 것을 의미한다. 

따라서 $t_{h}^{p}$은 pre-neuron의 스파이크 되는 시점을 뜻하고, 위에서 말한 디랙-델타 함수를 통해,
$G_h(t)$는 preneuron 스파이크가 되는 시점을 알 수 있다.

더욱 이해가 되기 위해 예시를 들어볼 수 있다.  

예를 들어, h번째 뉴런이 관측한 10ms에서 3ms와 6ms 지점에서 preneuron이 스파이크 되었다고 가정을 하면, $G_h(t)$는 $t=3ms, 6ms$일 때를 제외하고는
$G_h(t)$의 값은 0이다. 그렇다면 $G_h(3)$의 값은 어떻게 되는가? 정확히는 값을 따질 수 없으므로 0이 아니다라고 생각하고 넘기면 된다.
    
$$\begin{align}x_{h}~=~{1\over K}\int_{0}^{T} G_{h}(t^{\prime})\, dt^{\prime}\end{align}$$  

여기서 $x_{h}$는 스파이크 횟수를 나타낸다. 다만, $K$를 통해 정규화를 진행한 것을 의미하며, 디랙-델타함수의 합으로 표현된 $G_{h}(t)$를 정확하게
계산하기 위해서는 적분을 해야 한다.   

그래서 처음 시간인 $0ms$와 관측한 시간 $T ms$까지 적분을 하여, 그 사이에 스파이크가 되었으면, +1이 되어서 결과적으론 관측한 시간동안의 스파이크 횟수를 나타낸다.  

여기서 $K$는 관측시간 $Tms$ 동안 일어날 수 있는 최대 스파이크 수이다. 최대 스파이크 수를 구하기 위해서는 스파이크에 대한 약간의 이해가 필요한데, 

스파이크 되고 전위가 휴지 된 후, 스파이크가 다시 일어나기 위해서는 약간의 시간이 필요하다.  

따라서 그 시간을 고려하여 나온 최대 스파이크 수이다.

$$\begin{align}U(t) = \sum_{h}w_{h}(\int_{t-\alpha^{+}}^{t}\sum_{t_{h}^{p}}\delta(t^{\prime}-t_{h}^{p})\, dt^{\prime})\end{align}$$  

식 (6)에서는 post-neuron의 membrane potential을 계산한 것이다. 

정확히는 $t$시점에 post-neuron이 spike가 일어나고, 그 바로 직전에 스파이크가
일어나는 지점이 $t-\alpha$라고 볼 때 그 사이에 memebrane의 전위는 $\theta$를 넘기지 않는다.

post-neuron의 membrane potential이 올라가는 순간은 pre-neuron의 potential이 threthold를 넘겨서 스파이크 되고 그에 대한 영향으로
post-neuron의 membrane potential이 올라간다. 

그래서 식을 보면, pre-neuron의 스파이크 되는 시점인, $t_{h}^{p}$이 사용된다. 

최종 계산에서 potential은 위에 나오는 것과 같이 계산이 되며, 단순하게 정의한 식 (3a) 버전을
post-neuron에 대해서도 적용시킨 것으로 $w_{h}$는 각 가중치로 뉴런마다 특정 가중치를 곱하여 potential을 결정한다.

$$\begin{align}U^{tot} = \hat{y} = \sum_{t^{f}\in {r(t)=1}}U(t^{f})\end{align}$$

식 (7)은 post-neuron이 스파이크 되는 시점에서의 전위를 계산한 값이다. 

스파이크가 일어났다는 뜻은 potential이 threthold를 넘겼다는 것을 의미하며, 식 (6)을 통해 spike가 일어나기 전까지 post-neuron의 potential을 계산 할 수 있다.

따라서, $U^{tot}$은 post-neuron의 threthold를 넘긴 potential의 값의 합으로 표현된다. 

그리고 threthold를 넘긴 순간 potential은 휴지 potential이 되기 때문에, $U^{tot}$은 동일한 값들의 합으로 표현된다. 

즉, 예를 들어, threthold $\theta$가 10으로 설정되어 있을 때, 10을 넘기는 순간 spike가 일어나기 때문에 10의 값을 취할 뿐이다.

따라서 $U^{tot}$은 postsynaptic-neuron spike 횟수인 $R$과 연관이 있음을 알 수 있다.

$$\begin{align}f(\hat{y}) = \begin{cases} R = \gamma \hat{y} & \hat{y} > \theta \\ 0 & otherwise \end{cases}\end{align}$$

따라서 식 (8)과 같이 $U^{tot}$은 $R$과 비례 관계에 놓여있고, 여기서 $\gamma$는 비례 상수 역할로써, $T$에 비례하고 threthold인 $\theta$와
반비례 한다. 따라서 $\gamma \propto T \cdot \theta ^{-1}$ 라는 식이 만족된다.

여기서 선형관계로 표현된 activation function은 기존 ReLU function에서 $x$가 양수 방향으로 $\theta$만큼 이동하고,
기울기가 기존 ReLU의 $\gamma$ 배 라는 것으로 유사하다는 것을 입증할 수 있음.

Figure 1은 SNN의 activation function과 그것의 도함수가 나옴.

위 증명들을 통해, SNN의 activation function이 ReLU activation function과 유사하다는 것 점을 알 수 있고, 

이를 통해 ReLU에 적용된 방식을 새로운 스파이크 기반 learning rule에 적용하여 개발할 수 있게 됨.   

다음 section에서는 이를 통해 IF 뉴런에 적용되는 STDP-base backpropagation rule을 제안.  
## 2.2 Backpropagation using STDP
 여기서는 ReLU 활성화함수를 사용하는 신경망에서의 backpropagation update rule에서 영감을 받아, STDP를 사용하였음. 

 Figure 2에서 전통적인 신경망과 SNN의 네트워크 구조와 파라미터를 볼 수 있음.  

 이 두 네트워크의 주요 차이점은 데이터 통신 방식에 있음. 전통적인 신경망(left)은 실수를 입력 및 출력으로 처리하며, 
 SNN은 $T$ ms 시간 간격 내에서 spike train을 입력 및 출력으로 처리.  
 
 GD를 사용하는 ANN은 목표값 $d$ 와 출력값 $o$ 간의 차이 제곱를 최소화하는 문제로 해결.  
 
 $M$개의 output neuron이 $N$개의 training sample을 받을 때, 일반적인 loss function은 다음과 같음.  
 
 $$\begin{align}E = {1\over N} \sum_{k=1}^{N}\sum_{i=1}^{M}{(d_{k,i}-o_{k,i})}^2\end{align}$$

 위 일반적인 공식에서 GD와 학습률 $H$개의 입력을 받으면서 single training sample을 통한 선형 출력 neuron $i$의 경우, 가중치 변화 공식은 다음과 같다.

$$\begin{align}E={(d_{i}-o_{i})}^2={(d_{i}-\sum_{h}o_{h}w_{ih})}^2\rightarrow\frac{\partial E}{\partial w_{ih}}=-2(d_{i}-o_{i})\cdot o_{h}\end{align}$$

 이를 통해 $E~=~{(d_{i}-o_{i})}^2$라는 식을 가중치 update 수행할 수 있다.

$$\begin{align}\Delta w_{ih}~\propto~-\frac{\partial E}{\partial w_{ih}}~\rightarrow~{\Delta}w_{ih}={\mu}(d_{i}-o_{i})o_{h} \end{align}$$

여기서 이 식의 $d_{i}$, $o_{i}$, $o_{h}$은 ANN에서의 뉴런들의 Index이다. 이것을 spike train의 spike 수인 $L_{i}$, $G_{i}$, $G_{h}$로
각각 바꿀 수 있다. 

이것은 [35]에서 따온 것이다. 

그럴 경우 위에서 정의한 가중치 변화 공식은 SNN에서 synaptic weight update를 계산하도록
재구성할 수 있다.

이 가정은 Spiking IF neuron을 ReLU 뉴런으로 근사하는 Theorem 1에 따라 유효하다.

즉, 다시 말해 위에서 2.1에서 쓰인 증명을 통해 ANN의 weight change 공식을 SNN의 synaptic weight update 공식으로 근사할 수 있음.

SNN에서 weight update rule은 다음과 같음.

Post-synaptic neuron Spike Train : $$\begin{align}G_{i}(t)=\sum_{t_{i}^{p}{\in}\{r_{i}(t)=1\}}\delta(t-t_{i}^{p})\tag{12a}\end{align}$$

Desired neuron Spike Train : $$\begin{align} L_{i}(t) = \sum_{t_{i}^{q}{\in}\{z_{i}(t)=1\}}\delta(t-t_{i}^{q}) \tag{12b} \end{align}$$

그로 인해 얻어지는 weight update는 0부터 관측 시간 $T$까지의 적분을 통해 weight update량을 공식화할 수 있음.

$$\begin{align}\\
{\Delta}w_{ih} = {\mu}\int_{0}^{T}(L_{i}(t^{\prime})\,dt^{\prime}{\cdot}\int_{0}^{T}G_{h}(t^{\prime})\,dt^{\prime}
\end{align}
$$

하지만 식 (13)은 시간적인 측면에서 local하지 않고 관측 시간 $T$ 전부를 다룸. 

따라서 이것을 localizing 하기 위해, 시간 간격 $T$를 세분화함.

세분화하는 방식은 $T$를 각 시간 간격에 하나 또는 0개의 스파이크만 포함하도록 유도함.

이렇게 하는 이유는 뉴런의 스파이크가 일어나면 1, 스파이크가 일어나지 않으면 0으로 되기 때문에, Spike 하는 것에 타당성을 위해 이렇게 세분화함.

따라서 식 (13)을 세분화한 learning rule은 식 (14)으로 구체화함.


$$\begin{align}{\Delta}w_{ih}(t)~{\propto}~{\mu}(z_{i}(t)-r_{i}(t))s_{h}(t)\end{align}$$

위 공식을 구현하기 위해 event-based STDP와 anti-STDP를 combination했다.

이 방식은 teacher signal를 사용해, STDP와 anti-STDP 사이를 전환하며 synapse weight를 update함.

즉, 목표가 되는 neuron은 STDP를 따르며, 목표가 아닌 neuron은 anti-STDP를 따름.

우리가 desire한 spike train $z$는 입력 라벨에 따라 정의가 되는데, 목표 neuron은 최대 스파이크 빈도를 가지는 ($\beta$)를 통해 나타내고, 비목표 타겟은 silent함.

추가적으로 learning rule은 desired spike time $z_{i}(t)$에서 작동함. 즉, 모든 목표 뉴런에서 desire하는 spike time은 동일한 time임.

식 (15)은 supervised SNN의 출력 layer에 적용한 weight update 식이다.

$$\begin{align}{\Delta}w_{ih}(t) = {\mu}~{\cdot}~{\xi}_{i}(t) \sum_{t^{\prime}=t-{\epsilon}}^{t}s_{h}(t^{\prime})\end{align}$$

여기서 쓰이는 $\xi_{i}(t)$는 식 (16)으로 정의된다. 

$$\begin{align}{\xi_{i}(t)=\begin{cases} 1, & z_{i}(t) = 1,~r_{i} \ne 1~in~[t-{\epsilon},t] \\ -1, & z_{i}(t) = 0,~r_{i} = 1~in~[t-{\epsilon},t] \\ 0, & otherwise \end{cases}}\end{align}$$

식 (16)을 해석하자면, 목표가 아닌 뉴런의 시점 $T$에서 스파이크 되고,실제 pre-neuron의 시점 $T$에서 스파이크 되는경우, -1로 정의됨.  
만약 목표 뉴런의 시점 $T$에서 스파이크 되고, pre-neuron의 시점 $T$에서 스파이크가 되지 않을 경우, 1로 정의됨.

따라서 최종적으로 출력 layer의 synaptic weight는 식 (17)에 따라 이뤄진다.

$$\begin{align}w_{ih}(t) = w_{ih}(t) + {\Delta}w_{ih}(t)\end{align}$$

목표 neuron은 $z_{i}(t)에 의해 결정되는데, $z_{i}(t)=1$은 목표 neuron을 의미하고 $z_{i}(t)=0$은 비목표 neuron을 의미함.

출력 layer의 weight change는 desired spike time에서 이행됨.

Desired spike time $t$에서 목표 neuron은 STDP window로 알려진 매우 짧은 시간 간격 $[t-\Delta,t]$ 안에 이뤄줘야 한다.

그렇지 않은 경우, synaptic weight는 STDP window에서 pre-synaptic neuron 활동(주로 스파이크가 일어나냐 안 일어나냐에 따른)에 비례해 증가함.

presynaptic 활동은 STDP window에서 presynaptic의 spike 횟수를 나타내는 $\sum_{t^{\prime}=t-\epsilon}^{t}(s_{h}(t^{\prime}))$으로 표현됨.

반면에, 비목표 neuron이 발화되면 동일한 방식으로 weight depression가 일어남. 

이 방법은 전통적인 Gradient descent에서 유래하며, SNN에서 시공간적이고 local learning을 지원한다.

식 (18)에서는 ReLU neuron이 은닉층에 적용되는 backpropagation weight update rule이다. 

$$\begin{align}{\Delta}w_{hj} = \mu~\cdot~(\sum_{i} \hat{\xi}_{i}w_{ih})~\cdot~o_{j}~\cdot~[o_{h} > 0] \end{align}$$

$\hat{\xi_{i}}$은 desired와 출력 value 사이 차이읜 $(d_{i}-o_{i})$을 나타낸다.

그리고 우리의 SNN, $\hat{\xi_{i}}$는 식 (16)에 따라 \xi_{i}로 근사된다. 

value $[o_{h} > 0]$은 은닉층의 ReLU 뉴런의 도함수로 표현된다. 

IF neuron을 ReLU reuron으로 근사하면 (식 (8)을 통해), 다층 SNN의 스파이크 횟수를 기준으로 

$$\begin{align}{\Delta}w_{hj} = \mu \int_{0}^{T}(\sum_{i}\xi_{i}(t^{\prime})w_{ih}(t^{\prime}))\, dt^{\prime} \cdot \int_{0}^{T}(\sum_{t_{j}^{p}}\delta(t^{\prime}-t_{j}^{p}))\, dt^{\prime} \cdot ([\int_{0}^{T}\sum_{t_{h}^{p}}\delta(t^{\prime}-t_{h}^{p})\, dt^{\prime}] > 0)\end{align}$$

시간 $T$를 세부 간격 $[t-\epsilon,~t]$으로 나눈 후, 은닉층의 synaptic weight는 시간 측면에서 local rule을 따라 Update함.

$$\begin{align}{\Delta}w_{hj}(t) =  \left \lbrace \begin{align*}&{\mu} \cdot \sum_{i}\xi_{i}(t)w_{ih}(t) \cdot \sum_{t^{\prime}=t-{\epsilon}}^{t}s_{j}(t^{\prime}),~s_{h} = 1~in~[t-{\epsilon},t] \\ &0,~otherwise \end{align*} \right. \end{align}$$

최종적으로, 은닉층의 synaptic weight는 식 (21)에 의해 update됨.

$$\begin{align}w_{hj}(t) = w_{hj}(t) + {\Delta}w_{hj}(t) \end{align}$$

위의 learning rule은 은닉층의 neuron h가 발화하면 (postsynaptic neuron에서 spike가 발생하면) 0이 되지 않는다.

따라서, weight는 presynaptic($s_{j}(t)$)와 postsynaptic($s_{h}(j)$) spike time에 따라 업데이트됨. 그리고 이것은 표준 STDP 규칙을 따름.

추가적으로 ReLU의 도함수인 $o_{h} > 0 $은 식 (20)의 조건과 같이 IF neuron에서 spike generation과 유사함.

이 weight change rule에 따라 우리는 BP-STDP라고 불리는 STDP를 기반으로 Backpropagation algorithm를 따라는 다층 SNN을 만들었음.

Figure 3은 하나의 은닉층을 가진 SNN을 적용한 BP-STDP를 나타냄.

# 3. Result
제안한 모델을 XOR 문제와, iris dataset, MNIST dataset 이렇게 3개의 다른 문제들에 대해서 실험을 진행하였음.

실험에서 사용된 parameter는 Table 1에 따라 이뤄졌음. 

모든 실험에서 synaptic weight 초기화는 평균 $0$이고 표준편차가 $1$인 Gaussian function에서 랜덤적으로 숫자를 지정했음.

## 3.1 $XOR~problem$
BP-STDP 알고리즘은 XOR problem를 통해 선형적으로 분리되지 않는 문제를 해결할 수 있는 능력을 가졌다는 것을 보여주기 위해 실시되어짐.

dataset은 {(0.2, 0.2), (0.2, 1), (1, 0.2), (1, 1)}인 4가지의 datapoint를 포함하며, 각 point에 맞게 {0, 1, 1, 0}으로 매핑됨.

우리는 IF neuron 활성화를 위해 (즉, spike 발생을 위해) datapoint의 실수값인 0.2를 0으로 변경하였음.

신경망 구조는 2개의 입력, 20개의 은닉층 뉴련, 2개의 출력 IF neuron으로 구성하였다.

이 문제에선 은닉층의 neuron의 수는 결과에 큰 영향을 끼치지 않음.

대신 논문에서는 MNIST 분류 task에서 hidden reuron 수 영향을 조사함.

각 입력 neuron에서 spike train은 그에 연관된 입력 값에 따라 발화함.

예를 들어 value 1의 경우, 스파이크 일어났다는 의미로 보며 최대 스파이크 비율(250Hz)을 가진다고 표현됨.

Figure 4는 학습과정을 볼 수 있는데, 각 상자는 4가지의 입력 spike pattern에 대해 두 개의 출력 neuron 활동을 나타냄.

약 150 번의 학습 iteration 이후, 출력 뉴런은 입력에 따라 선택적으로 반응하게 됨.

Figure 5는 식 (22)로 정의된 에너지 함수를 사용하면서 learning 수렴 과정을 나타냄.

그 Figure에 따라, 적절한 learning rate($\mu$)는 $[0.01, 0.0005]$를 뜻함.

$$\begin{align}MSE = {1 \over N} \sum_{k=1}^{N}{({1 \over T} \sum_{t = 1}^{T}\xi^{k}(t))}^{2} ,\xi^{k}(t)=\sum_{i} \xi_{i}^{k}(t)
\end{align}$$

식 (22)에서, $N$와 $\xi_{i}^{k}(t)$은 각각 학습 batch size와 $k$개의 sample에 대한 출력 neuron $i$의 오차 값이다.

## 3.2 $Iris~dataset$
 Iris dataset은 (Setosa, Versicolour, Virginica)라는 3가지의 다른 꽃 유형을 가지고 있으며, petal과 sepal의 길이와 넓이를 나타냄. (즉 4가지 특징을 지님).

 각 특징 값을 0과 1 사이로 정규화해놓고, 입력 스파이크 train을 주축함.
 
 이 실험에서 신경망 구조는 4개의 입력, 30개의 은닉층, 3개의 출력 IF neuron으로 구성됨.

초기 실험에서는 10개 이상의 은닉층 뉴런을 사용하는 것이 정확도를 크게 향상시키지 않았음.

XOR 문제의 결과 유사하게 Figure 6은 학습을 통해 SNN이 세 가지 꽃 패턴에 대해 선택하여 반응하게 됨.

Figure 7은 식 (22)로 정의된 MSE를 기준으로 SNN의 학습 과정을 나타냄. 

최종적으로 5개의 fold cross validation 결과는 96% 정확도를 보여주며, 전통적인 Backpropagation을 사용하는 신경망은 96.7%의 정확도를 보여줌.

이 결과 BP-STDP 알고리즘이 시간적 SNN을 성공적으로 학습시킬 수 있음을 보여줌.

게다가 Table 2에서 BP-STDP를 다른 spike-based과 전통적인 supervised learning과 비교하였음.

BP-STDP는 다른 시공간적 GD-based approach보다 cost 효율성에서 우위 있으며, 대부분의 기존 방법보다 더 낫거나 동일한 성능을 보여줌.

## 3.3 $MNIST Dataset$

위 문제들보다 더 복잡한 문제에도 알고리즘을 적용하고 해결되는 지 평가하기 위해 MNIST dataset에 대한 SNN 구조는 784개의 입력 neuron과,
100부터 1500까지의 은닉층 neruon과 10개의 출력 IF neuron으로 구성된 BP-STDP이다.

SNN은 6만 개의 training sample에서 training하고, 만 개의 testing sample에서 testing sample에서 testing했음.

입력 spike train은 정규화된 pixel 값 ($[0,1]$ 범위)에 비례하여 생성된 스파이크ㄱ율과 랜덤 지연(random lags)를 사용해 생성했음.

Figure 8은 랜덤적으로 선택된 digit로부터 생성된 spike train에 대해 학습 후 출력 neuron의 membrane potential을 보여줌.

목표 neuron의 membrane potential은 빠르게 상승하여 threthold에 도달하고, 이 때의 다른 뉴런들의 활동(potential)은 거의 0에 근접함.

이 빠른 응답 ($< 9 ms$)는 신경망의 응답 지연을 줄임.

Figure 9a는 1200 학습 epoch로 학습된 과정을 볼 수 있음. 각 epoch는 50개의 MNIST digit을 나타냄.

이 그래프에서 MSE 차트는 학습률이 0.001과 0.0005일 때, 가장 빠르게 수렴하는 것을 볼 수 있음.

Figure 9b는 학습에서 1000개의 은닉층 neuron을 가진 SNN에서 정확도와 MSE 값을 볼 수 있음.

100과 900 epoch가 지난 후, 각각 성능은 90%와 96% 달성한 것을 볼 수 있음.

은닉층의 neuron의 수의 영향을 조사하기 위해 100부터 1500개의 은닉층 IF neuron을 가지는 6개의 SNN에 BP-STDP를 적용하였음.

Figure 10은 이 SNN에서 학습에 따른 정확도를 볼 수 있음. 최고 정확도는 500개 이상의 은닉층 neuron을 가지는 신경망이었음.

최종적으로 BP-STDP 알고리즘은 2개의 layer 또는 3개의 layer를 가지는 SNN에서 평가되었음.

SNN 구조(정확히는 은닉층 neuron의 수)는 더 나은 비교를 측정하기 위해 [39]에서 사용된 신경망 구조와 동일하게 하였음.

BP-STDP는 2개의 layer SNN에서는 $96.6 \pm 0.1%$을 달성하였으며, 3개의 layer SNN에서는 $97.2 \pm 0.07 %$을 달성하였음.

이 결과는 backpropagation으로 학습한 전통적인 신경망과 유사한 정확도를 나타냄.

Table 3은 제안된 supervised learning(BP-STDP)와 전통적인 backpropagation 신경망(또는 GD), Deep SNN에 적용된 spiking backpropagation 방법과
최근 MNIST 분류로 사용된 STDP-based SNN을 비교하였음.

이 비교는 시간적 SNN 구조에 적용하고 생물학적으로 영감받은 BP-STDP의 성공을 입증함.

BP-STDP는 다층 SNN에 적용한 end-to-end STDP based, supervised learing approach임.

[40,41]에서 소개된 SNN은 특징 추출을 위해 다층 STDP 학습을 발전사켰지만, 최종적인 supervised layer에서 SVM 분류기를 사용하였음.

비록 GD approach가 성공적이지만, 전력 효율적인 STDP learning과 다르게 생물학적 영감을 제공하지 않음. 

특히 최근에는 backpropagation을 사용하는 Deep SNN이 BP-STDP의 효율보다 더 높은 효울성을 갖고 옴

하지만 이것은 neuron의 스파이크 event를 사용하는 것 대신에 그들의 membrane potential을 사용하여 활상화 함수의 도함수를 계산함.

# 4. Discussion

BP-STDP는 SNN의 novel한 supervised learning으로 소개함.

그것은 최첨단 전통적인 GD approach와 비교할만 성능 보장을 보여줌.

BP-STDP는 생물학적 영감을 제공하는 local learning을 하는데, local learning은 IF neuron의 다음 layer에서 spike rate 뿐만 아니라 spike time까지 고려하는 방법임.

Bengio et al.에서 synaptic weight update가 presynaptic spike event와 postsynaptic 시간적 활동이 비례한다는 것을 보여줌.

그리고 이것은 STDP 규칙과 유사하며, STDP가 postsynaptic 시간적 spike rate와 연관되어 있다는 Hioton의 생각을 확인함.

제안된 알고리즘은 ReLU neuron이 포함된 전통적인 신경망에서 사용된 backpropagation rule에서 영감을 받았음.

하지만 이것은 생물학적으로 가소성하고 시간적으로 local learning rule을 개발함.

이러한 접근은 SNN에서 spike based 구조를 지원하기 위해 IF neuron을 ReLU neuron으로 근사하여 초기에 이뤄졌음.

spiking supervised learning은 시간적으로 presynaptic와 postsynaptic neuron spike event에 대응되는 spiking neural layer에 적용되는 STDP와 anti-STDP를 혼합하여 제공함.

따라서, 이 구조에서는 시간적으로 local한 STDP의 효율성과 GD의 정확도 모두에 대해 이점이 있음.

주된 질문은, 다층 신경망 구조에서 IF neuron의 spiking event에 대해 오차 propagation과 어떻게 연관되었는가?이다.

이 질문에 답하기 위해 오차 값을 뉴런을 발화하거나$(\sum_{i}\xi_{i} > 0)$ 억제하는 신호$(\sum_{i}\xi_{i} <> 0)$로 가정해봤음.

이 때 뉴런을 자극(억제)하는 것은 synapse의 input weight와 presynaptic spike event에 의해 비례하여 통제되는 membrane potential를 증가(감소)시키는 것을 의미함.

따라서 BP-STDP update rule은 synaptic weight가 시간적으로 오차 신호와 presynaptic spike time에 따라 변화하며, 각 시간 단계의 은닉층의 행동 potential을 조작함.

실험 결과는 다층 SNN에 supervised learning을 구현함에 있어 BP-STDP의 성공을 보여줌.

XOR problem은 BP-STDP가 spike train으로 표현된 선형적으로 분리되지 않는 sample을 시간 간격 $T$ ms 시간 간격 동안 분류할 수 있는 능력을 입증했음.

IRIS와 MNIST 분류와 같이 복잡한 문제는 전통적인 backpropagation 알고리즘과 최근 SNN에 비교할만한 성능을 내보낸 것을 보여주며, BP-STDP가 spiking pattern classification에 의해 end-toend STDP-based supervised learning을 제공함.

생물학적으로 영감을 받고 시갖거으로 local한 STDP rule은 뇌에서 발생하는 효율적인 계산에 한 걸음 더 가까워지게 함.

우리의 지식 안에서는, 이 접근법이 cost적으로 비용이 높은 GD를 피하면서 고성능 STDP-based supervised learning을 제공하는 최초의 방법임.

# 5. Conclusion

이 논문은 만약 뉴런의 행동이 spike rate에 매핑되면 IF neuron이 ReLU에 근사할 수 있음을 보임.

따라서 spiking IF neuron의 신경망은 전통적인 신경망에 적용된 backpropagation learning와 비슷하다고 볼 수 있음.

IF neuron을 쓰는 다층 SNN에서 적용된 STDp와 antiSTDP의 조합으로 시간적이고 local한 learning rule을 제안했음.

BP-STDP는 spiking neuron에서 효율적인 STDP rule을 사용하여 생물학적 영감의 이점을 얻고, 다층 신경망을 학습하면서 GD의 힘을 얻었다.

또한 GD-based weight change rule을 spike-based STDP rule로의 변환은 spiking GD rule의 발전보다 더 쉽고, cost적으로 비용이 절약됨.

XOR 문제에서의 실험은 제안된 SNN가 선형적이지 않게 분리된 pattern을 분류할 수 있음을 보여줌.

그리고 Iris와 MNIST dataset에서의 최종 평가는 최첨단인 전통과 spiking neuron의 다층 신경망과 비교할 수 있는 높은 분류 정확도를 보여줌.

BP-STDP 모델의 결과는 BP-STDP와 적규화 module이 추가된 심층 SNN을 개발하기 위해 향후 연구의 필요성을 보여줌.

다층 SNN은 더 큰 pattern recongnition 작업에 활용할 수 있으며, 뇌와 유사한 효율적인 계산을 유지할 수 있음.