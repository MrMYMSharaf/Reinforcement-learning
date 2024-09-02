# What is Reinforcement Learning?

**Reinforcement learning** is an area of machine learning in which an agent learns by interacting with its environment to achieve a goal. The agent performs actions and receives feedback in the form of rewards or penalties. The goal is to learn a strategy, or policy, that maximizes cumulative rewards over time.

![Reinforcement Learning Image](https://via.placeholder.com/600x200.png)

## What is reinforcement learning?

Reinforcement learning (RL) is a machine learning (ML) technique that trains software to make decisions to achieve the most optimal results. It mimics the trial-and-error learning process that humans use to achieve their goals. Software actions that work towards your goal are reinforced, while actions that detract from the goal are ignored.

RL algorithms use a reward-and-punishment paradigm as they process data. They learn from the feedback of each action and self-discover the best processing paths to achieve final outcomes. The algorithms are also capable of delayed gratification. The best overall strategy may require short-term sacrifices, so the best approach they discover may include some punishments or backtracking along the way. RL is a powerful method to help artificial intelligence (AI) systems achieve optimal outcomes in unseen environments.

## Key Concepts

- **Agent**: The learner or decision maker.
- **Environment**: Everything the agent interacts with.
- **Actions**: The set of all possible moves the agent can make.
- **Rewards**: The feedback from the environment based on the agent's actions.

## How Does Reinforcement Learning Work?

1. **Initialization**: The agent is initialized with no prior knowledge.
2. **Interaction**: The agent interacts with the environment by taking actions.
3. **Feedback**: The environment provides feedback in the form of rewards or penalties.
4. **Learning**: The agent updates its policy based on the feedback to improve future performance.

## What are the benefits of reinforcement learning?

There are many benefits to using reinforcement learning (RL). However, these three often stand out:

1. **Excels in complex environments**: RL algorithms can be used in complex environments with many rules and dependencies. In the same environment, a human may not be capable of determining the best path to take, even with superior knowledge of the environment. Instead, model-free RL algorithms adapt quickly to continuously changing environments and find new strategies to optimize results.

2. **Requires less human interaction**: In traditional ML algorithms, humans must label data pairs to direct the algorithm. When you use an RL algorithm, this isn’t necessary. It learns by itself. At the same time, it offers mechanisms to integrate human feedback, allowing for systems that adapt to human preferences, expertise, and corrections.

3. **Optimizes for long-term goals**: RL inherently focuses on long-term reward maximization, which makes it apt for scenarios where actions have prolonged consequences. It is particularly well-suited for real-world situations where feedback isn't immediately available for every step, since it can learn from delayed rewards.

## What are the use cases of reinforcement learning?

Reinforcement learning (RL) can be applied to a wide range of real-world use cases. Here are some examples:

- **Marketing personalization**: In applications like recommendation systems, RL can customize suggestions to individual users based on their interactions, leading to more personalized experiences. For example, an application may display ads to a user based on some demographic information. With each ad interaction, the application learns which ads to display to the user to optimize product sales.

- **Optimization challenges**: Traditional optimization methods solve problems by evaluating and comparing possible solutions based on certain criteria. In contrast, RL introduces learning from interactions to find the best or close-to-best solutions over time. For example, a cloud spend optimizing system uses RL to adjust to fluctuating resource needs and choose optimal instance types, quantities, and configurations.

- **Financial predictions**: The dynamics of financial markets are complex, with statistical properties that change over time. RL algorithms can optimize long-term returns by considering transaction costs and adapting to market shifts. For instance, an algorithm could observe the rules and patterns of the stock market before it tests actions and records associated rewards.

## How does reinforcement learning work?

The learning process of reinforcement learning (RL) algorithms is similar to animal and human reinforcement learning in the field of behavioral psychology. For instance, a child may discover that they receive parental praise when they help a sibling or clean but receive negative reactions when they throw toys or yell. Soon, the child learns which combination of activities results in the end reward.

An RL algorithm mimics a similar learning process. It tries different activities to learn the associated negative and positive values to achieve the end reward outcome.

### Key concepts

In reinforcement learning, there are a few key concepts to familiarize yourself with:

- **Agent**: The ML algorithm (or the autonomous system).
- **Environment**: The adaptive problem space with attributes such as variables, boundary values, rules, and valid actions.
- **Action**: A step that the RL agent takes to navigate the environment.
- **State**: The environment at a given point in time.
- **Reward**: The positive, negative, or zero value—in other words, the reward or punishment—for taking an action.
- **Cumulative reward**: The sum of all rewards or the end value.

### Algorithm basics

Reinforcement learning is based on the Markov decision process, a mathematical modeling of decision-making that uses discrete time steps. At every step, the agent takes a new action that results in a new environment state. Similarly, the current state is attributed to the sequence of previous actions.

Through trial and error in moving through the environment, the agent builds a set of if-then rules or policies. The policies help it decide which action to take next for optimal cumulative reward. The agent must also choose between further environment exploration to learn new state-action rewards or select known high-reward actions from a given state. This is called the exploration-exploitation trade-off.

## What are the types of reinforcement learning algorithms?

There are various algorithms used in reinforcement learning (RL)—such as Q-learning, policy gradient methods, Monte Carlo methods, and temporal difference learning. Deep RL is the application of deep neural networks to reinforcement learning. One example of a deep RL algorithm is Trust Region Policy Optimization (TRPO).

All these algorithms can be grouped into two broad categories:

1. **Model-based RL**: Typically used when environments are well-defined and unchanging and where real-world environment testing is difficult.

    - The agent first builds an internal representation (model) of the environment. It uses this process to build this model:
        - It takes actions within the environment and notes the new state and reward value.
        - It associates the action-state transition with the reward value.
    - Once the model is complete, the agent simulates action sequences based on the probability of optimal cumulative rewards. It then further assigns values to the action sequences themselves. The agent thus develops different strategies within the environment to achieve the desired end goal.
    
    - **Example**: Consider a robot learning to navigate a new building to reach a specific room. Initially, the robot explores freely and builds an internal model (or map) of the building. For instance, it might learn that it encounters an elevator after moving forward 10 meters from the main entrance. Once it builds the map, it can build a series of shortest-path sequences between different locations it visits frequently in the building.

2. **Model-free RL**: Best to use when the environment is large, complex, and not easily describable. It’s also ideal when the environment is unknown and changing, and environment-based testing does not come with significant downsides.

    - The agent doesn’t build an internal model of the environment and its dynamics. Instead, it uses a trial-and-error approach within the environment. It scores and notes state-action pairs—and sequences of state-action pairs—to develop a policy.
    
    - **Example**: Consider a self-driving car that needs to navigate city traffic. Roads, traffic patterns, pedestrian behavior, and countless other factors can make the environment highly dynamic and complex. AI teams train the vehicle in a simulated environment in the initial stages. The vehicle takes actions based on its current state and receives rewards or penalties.

## What is the difference between reinforced, supervised, and unsupervised machine learning?

While supervised learning, unsupervised learning, and reinforcement learning (RL) are all ML algorithms in the field of AI, there are distinctions between the three.

### Reinforcement learning vs. supervised learning

In supervised learning, you define both the input and the expected associated output. For instance, you can provide a set of images labeled dogs or cats, and the algorithm is then expected to identify a new animal image as a dog or cat.

Supervised learning algorithms learn patterns and relationships between the input and output pairs. Then, they predict outcomes based on new input data. It requires a supervisor, typically a human, to label each data record in a training data set with an output.

In contrast, RL has a well-defined end goal in the form of a desired result but no supervisor to label associated data in advance. During training, instead of trying to map inputs with known outputs, it maps inputs with possible outcomes. By rewarding desired behaviors, you give weightage to the best outcomes.

### Reinforcement learning vs. unsupervised learning

Unsupervised learning algorithms receive inputs with no specified outputs during the training process. They find hidden patterns and relationships within the data using statistical means. For instance, you could provide a set of documents, and the algorithm may group them into categories it identifies based on the words in the text. You do not get any specific outcomes; they fall within a range.

Conversely, RL has a predetermined end goal. While it takes an exploratory approach, the explorations are continuously validated and improved to increase the probability of reaching the end goal. It can teach itself to reach very specific outcomes.

## What are the challenges with reinforcement learning?

While reinforcement learning (RL) applications can potentially change the world, it may not be easy to deploy these algorithms.

1. **Practicality**: Experimenting with real-world reward and punishment systems may not be practical. For instance, testing a drone in the real world without testing in a simulator first would lead to significant numbers of broken aircraft. Real-world environments change often, significantly, and with limited warning. It can make it harder for the algorithm to be effective in practice.

2. **Interpretability**: Like any field of science, data science also looks at conclusive research and findings to establish standards and procedures. Data scientists prefer knowing how a specific conclusion was reached for provability and replication.

    With complex RL algorithms, the reasons why a particular sequence of steps was taken may be difficult to ascertain. Which actions in a sequence were the ones that led to the optimal end result? This can be difficult to deduce, which causes implementation challenges.

## How can AWS help with reinforcement learning?

Amazon Web Services (AWS) has many offerings that help you develop, train, and deploy reinforcement learning (RL) algorithms for real-world applications.

- With Amazon SageMaker, developers and data scientists can quickly and easily develop scalable RL models. Combine a deep learning framework (like TensorFlow or Apache MXNet), an RL toolkit (like RL Coach or RLlib), and an environment to mimic a real-world scenario. You can use it to create and test your model.

- With AWS RoboMaker, developers can run, scale, and automate simulation with RL algorithms for robotics without any infrastructure requirements.

- Get hands-on experience with AWS DeepRacer, the fully autonomous 1/18th scale race car. It boasts a fully configured cloud environment that you can use to train your RL models and neural network configurations.

Get started with reinforcement learning on AWS by creating an account today.
