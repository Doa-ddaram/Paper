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
  
$$f(y) = max(0,~y),~y = \sum_{h} x_hw_h \qquad (1) $$   

$$\frac{\partial f}{\partial y} = \begin{cases} 1, & y > 0  \\ 0, & y \le 0 \end{cases} \qquad (2)$$

이론적으로 IF neuron은 ReLU neuron을 근사할 수 있음. 특히, IF 뉴런의 membrane potential은 ReLU neuron의 활성화 값으로 근사 가능.

이것을 증명하기 위해서 LIF가 아닌 IF 뉴런은 $U(t)$라는 membrane potential이 임계값인 $\theta$ 를 넘기면 spike train에 흔적이 남고 fire하게 된다.  

$$ U(t) = U (t- \Delta t)~+~\sum_{h}w_{h}(t)s_{h}(t) \qquad (3a)$$  
$$if~U(t)~\ge~\theta~$ then $ r(t)~=~1,~U(t)~=~U_{rest} \qquad (3b)$$  

여기서 $s_h(t)$와 $r_h(t)$는 $t$ 시간에 대해 시냅스적으로 pre와 post neuron의 spike를 각각 의미한다. 따라서 $s_{h}(t),r_{h}(t) \in \{0,~1\}$ 이다.
추가적으로 아래 첨자로 표시되는 h는 h번째 시냅스? 뉴런?을 의미함.  

neuron의 membrane potential은 fire할 때, 휴지(resting) potential인 $U_{rest}$으로 재설정됨. 이 때, $U_{rest}$은 0으로 생각함.  

$G_{h}(t)$은 시냅스에서 pre-neuron의 spike train을 의미하며, 여기서 사용되는 spike train은 앞으로도 계속 사용되는 개념으로

스파이크 되는 지점의 시점인 $t_{h}^p$의 정보를 갖고 있다고 보면 된다. 그리고 여기서 사용되는 $G_{h}(t)$는 디랙-델타 함수의 합으로 표현되는데,
그 이유는 적분을 할 때, $\delta(t-t_{h}^{p})$를 유의미한 값을 가지기 위해서다.  

디랙-델타 함수는 $\delta$의 함수 꼴로 표현되는데, $\int_{-\infty}^{\infty} \delta(t)\,dt = 1 $,여기서 $t$가 $t_{h}^{p}$와 일치하지 않는다면,

$\delta(t-t_{h}^{p})$가 0이 된다. 이를 통해 스파이크 되는 지점을 잡아서, 스파이크 되는 시점을 모아 놓은 함수를 얻을 수 있다. 그 내용이 식 (4)이다.  

$G_h(t)~=~\sum_{t_{h}^{p}\in {s_{h}(t)=1}}\delta (t-t_{h}^{p})\qquad (4)$  

여기서 $G_h(t)$는 spike train이다. 식을 분석하자면, $s_{h}(t) = 1$는 위에서 말한 대로 pre-neuron의 스파이크를 의미하며 $t$ 시점의 $h$번째 뉴런이
스파이크되었다는 것을 의미한다. 따라서 $t_{h}^{p}$은 pre-neuron의 스파이크 되는 시점을 뜻하고, 위에서 말한 디랙-델타 함수를 통해,
$G_h(t)$는 preneuron 스파이크가 되는 시점을 알 수 있다. 예시를 통해서 더욱 이해에 되움을 주겠다.  

예를 들어, h번째 뉴런이 관측한 10ms에서 3ms와 6ms 지점에서 preneuron이 스파이크 되었다고 가정을 하면, $G_h(t)$는 $t=3ms, 6ms$일 때를 제외하고는
$G_h(t)$의 값은 0이다. 그렇다면 $G_h(3)$의 값은 어떻게 되는가? 정확히는 값을 따질 수 없으므로 0이 아니다라고 생각하고 넘기면 된다.
    
$x_{h}~=~{1\over K}\int_{0}^{T} G_{h}(t^{'})\, dt^{'}\qquad (5)$  

여기서 $x_{h}$는 스파이크 횟수를 나타낸다. 다만, $K$를 통해 정규화를 진행한 것을 의미하며, 디랙-델타함수의 합으로 표현된 $G_{h}(t)$를 정확하게
계산하기 위해서는 적분을 해야 한다.   

그래서 처음 시간인 $0ms$와 관측한 시간 $T ms$까지 적분을 하여, 그 사이에 스파이크가 되었으면, +1이 되어서 결과적으론 관측한 시간동안의 스파이크 횟수를 나타낸다.  

여기서 $K$는 관측시간 $Tms$ 동안 일어날 수 있는 최대 스파이크 수이다. 최대 스파이크 수를 구하기 위해서는 스파이크에 대한 약간의 이해가 필요한데, 

스파이크 되고 전위가 휴지 된 후, 스파이크가 다시 일어나기 위해서는 약간의 시간이 필요하다.  

따라서 그 시간을 고려하여 나온 최대 스파이크 수이다.

$U(t) = \sum_{h}w_{h}(\int_{t-\alpha^{+}}^{t}\sum_{t_{h}^{p}}\delta(t^{'}-t_{h}^{p})\, dt^{'}) \qquad (6) $  

식 (6)에서는 post-neuron의 membrane potential을 계산한 것이다. 

정확히는 $t$시점에 post-neuron이 spike가 일어나고, 그 바로 직전에 스파이크가
일어나는 지점이 $t-\alpha$라고 볼 때 그 사이에 memebrane의 전위는 $\theta$를 넘기지 않는다.

post-neuron의 membrane potential이 올라가는 순간은 pre-neuron의 potential이 threthold를 넘겨서 스파이크 되고 그에 대한 영향으로
post-neuron의 membrane potential이 올라간다. 

그래서 식을 보면, pre-neuron의 스파이크 되는 시점인, $t_{h}^{p}$이 사용된다. 

최종 계산에서 potential은 위에 나오는 것과 같이 계산이 되며, 단순하게 정의한 식 (3a) 버전을
post-neuron에 대해서도 적용시킨 것으로 $w_{h}$는 각 가중치로 뉴런마다 특정 가중치를 곱하여 potential을 결정한다.

$U^{tot} = \hat{y} = \sum_{t^{f}\in {r(t)=1}}U(t^{f}) \qquad (7) $

식 (7)은 post-neuron이 스파이크 되는 시점에서의 전위를 계산한 값이다. 

스파이크가 일어났다는 뜻은 potential이 threthold를 넘겼다는 것을 의미하며, 식 (6)을 통해 spike가 일어나기 전까지 post-neuron의 potential을 계산 할 수 있다.

따라서, $U^{tot}$은 post-neuron의 threthold를 넘긴 potential의 값의 합으로 표현된다. 

그리고 threthold를 넘긴 순간 potential은 휴지 potential이 되기 때문에, $U^{tot}$은 동일한 값들의 합으로 표현된다. 

즉, 예를 들어, threthold $\theta$가 10으로 설정되어 있을 때, 10을 넘기는 순간 spike가 일어나기 때문에 10의 값을 취할 뿐이다.

따라서 $U^{tot}$은 postsynaptic-neuron spike 횟수인 $R$과 연관이 있음을 알 수 있다.

$$f(\hat{y}) = \begin{cases} R = \gamma \hat{y} & \hat{y} > \theta \\ 0 & otherwise \end{cases} \qquad (8)$$

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
 SNN은 $ T$ ms 시간 간격 내에서 spike train을 입력 및 출력으로 처리.  
 
 GD를 사용하는 ANN은 목표값 $d$ 와 출력값 $o$ 간의 차이 제곱를 최소화하는 문제로 해결.  
 
 $M$개의 output neuron이 $N$개의 training sample을 받을 때, 일반적인 loss function은 다음과 같음.  
 
 $E = {1\over N} \sum_{k=1}^N$