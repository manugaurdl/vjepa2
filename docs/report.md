## GOAL: write a report/summary of the project so far. It should self-contained, so that a reader can understand the project, its motivation, current state, and be able to effectively discuss future directions.

So far, I was exploring recurrence as a means to a persistent memory, so that it can enable long-video understanding/generation/planning.
Started with proxy task - SSV2, 8 uniformly sampled frames, temporal changes, trained for action classification.

Arch: Started with a simple additive RNN without state-decay

prior: decay is needed for long videos; for now just find a way to encode novel updates and maintain a persistent memory.

Findings: 
* Layernorm memory and updates
* Predictive coding is a better prior than using sigmoid gates; 
    - Easier to predict the next frame or directly predict new pixels (how to do in dino space / with semantics?) than learning negative weights to decay specific dimensions

Making predictive coding work:
    Error = f(x_t) - W_pred(S_{t-1})
    <!-- Precision = sigmoid(W(Error))
    S_t = S_{t-1} + Precision * Error -->
    S_t = S_{t-1} + W(Error)
    
    Loss = CE + L2(Error)

* f --> linear
* W_pred --> MLP
* Precision --> linear
* lambda=.1 ; better W_pred --> better memory dynamics

Potential future directions:
* WM memory ; Video Gen
* Planning: [Dreamer, TD-MPC2, DINO-WM, temporal Straightening]
* Understanding
* Robotics
    - keep the predictive coding inductive bias; add another decoder for policy
    - general purpose framework for visual memory; video generation can be an instantiation of this
    - with RL: can get dense supervision; quietStar; guide compression (explicity or implicitly through loss)

outperforming concat(frames)/transformer


- chunkwise frame --> local compression; multiple states
-minimize state size but also maximize state expressivity for future use;
similar to the reward/task in RL, here the data (VQA questions) directlys guide the compression

multi-staged training:
stage 1: learn a video encoder; a \sout{forward dynamics model}; within latent-space; model knows what is important and can recurrently ingest videos 
    - future prediction + local compression --> both can be posed as (VQA/latent pred)
stage 2: make it usable for downstream tasks; VQA - both long term and short term



--> babyAI; 
planning - retrieval + search

if stage 1 is general enough, can be used for task requiring:
* planning - make it action conditioned - what tasks??
* memory - navigation; object maybe be hidden inside cupboard

