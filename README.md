


# Question-Answering Dense Video Events
 Question-Answering Dense Video Event

## Introduction
Multimodal Large Language Models (MLLMs) have shown excellent performance in question-answering of single-event videos. In this paper, we present question-answering dense video events, a novel task that requires answering and grounding the dense-event questions in long videos, thus challenging MLLMs to faithfully comprehend and reason about multiple events occurring over extended time periods. To facilitate the study, we construct DeVE-QA – a dataset featuring 78K questions about 26K events on 10.6K long videos. We then benchmark and show that existing MLLMs excelling at single-event QA struggle to perform well in DeVE-QA. For improvement, we propose DeVi, a novel training-free MLLM approach that highlights a hierarchical captioning module, a temporal event memory module, and a self-consistency checking module to respectively detect, contextualize and memorize, and ground dense-events in long videos for question answering. Extensive experiments show that DeVi is superior at answering dense-event questions and grounding relevant video moments. Compared with existing MLLMs, it achieves a remarkable increase of 4.1% and 3.7% for G(round)QA accuracy on DeVE-QA and NExT-GQA respectively. 

<div align="center">
  <img width="100%" alt="Visually Grounded VideoQA" src="./img/framework.png">
</div>

## Result Visualization 
<div align="center">
  <img width="90%" alt="NExT-GQA for visually-grounded VideoQA" src="./img/sample2.png">
</div>
