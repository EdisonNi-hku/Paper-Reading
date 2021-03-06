# Paper-Reading
Paper reading list in natural language processing (NLP), with special emphasis on **Finance+NLP** and relevant topics. This repo will keep updating 🤗 ...

- [Claim Verification](#claim-verification)
    - [Scientific Claim Verification](#scientific-claim-verification)
    - [Fact-check of Other domain](#fact-check-of-othergeneral-domains)
    - [Data Collection](#data-collection)
- [Transfer Learning and Multi-task Learning](#transfer-learning-and-MTL)
    - [Adapter Based Models](#adapter-based-models)
    - [Other models](#other-models)
    - [MTL Analysis](#mtl-analysis)
- [NLP in Finance](#nlp-in-finance)
    - [Financial NLP Models](#financial-nlp-models)
    - [Financial NLP Tasks](#financial-nlp-tasks)
- [Numbers in NLP](#number-and-numeracy)
    - [Number Representation](#number-representation)
    - [Numeracy](#numeracy)
- [Relation Extraction](#relation-extraction)
- [Semantic Role Labeling](#semantic-role-labeling)
- [Dependency Parsing](#dependency-parsing)
- [Domain Adaptation](#domain-adaptation)
- [Transformers Interpretation](#transformers-interpretation)
- [NLP in Programming Language](#nlp-in-programming-language)

## Claim Verification

### Scientific Claim Verification
* **ClaimGenBART**: "Generating Scientific Claims for Zero-Shot Scientific Fact Checking" (Wright et al., 2022 ACL) [[PDF]](https://arxiv.org/pdf/2203.12990.pdf)
* **MultiVers**: "Improving scientific claim verification with weak supervision and full-document context" (Wadden et al., 2022 NAACL.findings) [[PDF]](https://arxiv.org/pdf/2112.01640.pdf): provided a nice summarization of background of SCV.
* **ARSJoint**: "Abstract, Rationale, Stance: A Joint Model for Scientific Claim Verification" (Zhang et al., 2021 EMNLP) [[PDF]](https://aclanthology.org/2021.emnlp-main.290.pdf): Another joint approach dealing with error propagation in pipeline. Similar to ParagraphJoint, contextualize the entire abstract with claim. Use hierachical attention to compute sentence attentions/abstract attention. Information sharing/MTL joint training enabled by complicated prediction headers.
* **ParagraphJoint**: "MULTIVERS: Improving scientific claim verification with weak supervision and full-document context" (Li et al., 2021 AAAI) [[PDF]](https://arxiv.org/pdf/2012.14500v1.pdf): joint train rationale selection, and stance prediction, which is different from the three-step pipeline. Does not outperform pipeline models. Benefit: encode full abstract(compact paragraph encoding) compared to "extract-then-label"
* **VERT5ERINI**: "Scientific Claim Verification with VERT5ERINI" (Pradeep et al., 2021 EACL Workshop) [[PDF]](https://aclanthology.org/2021.louhi-1.11.pdf): a pipeline approach based on T5-3B. Use MS-MACRO dataset for pre-finetuning.
* **SciFact**: "Fact or Fiction: Verifying Scientific Claims" (Wadden et al., 2020 EMNLP) [[PDF]](https://aclanthology.org/2020.emnlp-main.609.pdf)


### Fact-check of Other/General domains
* **KGAT**: "Fine-grained Fact Verification with Kernel Graph Attention Network" (Liu et al., 2020 ACL) [[PDF]](https://arxiv.org/pdf/1910.09796.pdf)
* **GEAR**: "GEAR: Graph-based Evidence Aggregating and Reasoning for Fact Verification" (Zhou et al., 2019 ACL) [[PDF]](https://aclanthology.org/P19-1085.pdf)
* **UKP Snopes**: "A richly annotated corpus for different tasks in automated factchecking" (Hanselowski et al., 2019 CoNLL)


### Data Collection
* **Presidential Debates**: "Detecting Check-worthy Factual Claims in
Presidential Debates" (2015) [[URL]](https://dl.acm.org/doi/10.1145/2806416.2806652): whether a sentence is "non-factual", "unimportant factual", and "check-worthy factual". Data collection process: debate transcripts -> filter sentences from president candidates -> discard short sentences -> Use a data collection website to annotate -> 140 annotators -> using screening sentences to select high-quality annotators.
* **ClaimBuster**: "A Benchmark Dataset of Check-Worthy Factual Claims" AAAI(2020) [[PDF]](https://ojs.aaai.org/index.php/ICWSM/article/view/7346/7200): Also on the domain of presidiential debates. Compared with previous work, it reduced bias, and improved time-period of data. Similar transcripts processing. Use 40 labeled sentences to train annotators, as well as on-site training workshops. Use screening sentences to detect labeling quality. Monetary rewards and score rank to encourage better annotation.
* **Judicial Decisions**: "Automated fact-value distinction in court opinions" (2020) [[PDF]](https://link.springer.com/epdf/10.1007/s10657-020-09645-7?author_access_token=0lrxR5amL26ii9rbxOyRRve4RwlQNchNByi7wbcMAY4Rn4AGeJ9qqiUyLFGlSyn90_9MSB1ZXV1_BuuMOQ4sUXyeLq83OpD7B678nRCUDq6T2yW5EWuYBLhb4CC82O6D5dt5Bflo8nVd86wC0_EaFA%3D%3D): classification over fact statements(fact about the case) & value statements(principles applicable to the facts). Fact/opinion classification in law domain is different from other domain. Data was collected by parsing, using labels in the section headers.
* **CheckThat!2019**: "Overview of the CLEF-2019 CheckThat! Lab: Automatic Identification and Verification of Claims. Task 1: Check-Worthiness" (2019) [[PDF]](http://ceur-ws.org/Vol-2380/paper_269.pdf): different from ClaimBuster, this work is based on annotations by a fact-checking organization: mark those claims whose factuality was challenged by the fact-checkers.
* **CheckThat!2020**: "Overview of CheckThat! 2020: Automatic Identification and Verification of Claims in Social Media" (2020) [[PDF]](https://arxiv.org/pdf/2007.07997.pdf): claim verification pipeline in twitter. check-worthiness -> verified claim retrieval -> supporting evidence retrieval -> claim verification. Check-worthiness data collection: define 5 questions about check-worthiness. If the answers are all possitive th tweet is annotated to worth-checking. 2-5 annotators independently, then discuss disagreement.
* **Covid Infodemic**: "Fighting the COVID-19 Infodemic: Modeling the Perspective of Journalists, Fact-Checkers, Social Media Platforms, Policy Makers, and the Society" EMNLP.Findings(Alam et al., 2021) [[PDF]](https://aclanthology.org/2021.findings-emnlp.56.pdf): three annotators per tweet. Resolve cases of disgreement in a consolidation discusion. Annotation instruction provided. The annotators are required to annotate 7 questions regarding a tweet. The questions help annotators to think more thoroughly, and provide comprehensive annotation.
* **Covid Infodemic Annotation Platform**: "Fighting the COVID-19 Infodemic in Social Media:A Holistic Perspective and a Call to Arms". AAAI(Alam et al., 2021) [[PDF]](https://ojs.aaai.org/index.php/ICWSM/article/view/18114/17917): crowd-sourcing annotation platform based on MicroMappers.


### Others
* **FRUIT**: "FRUIT: Faithfully Reflecting Updated Information in Text" (L. Logan IV., et al., 2021) [[PDF]](https://arxiv.org/abs/2112.08634)

## Transfer Learning and MTL
### Adapter Based Models
* **Hyperformer**: "Parameter-efficient Multi-task Fine-tuning for Transformers via Shared Hypernetworks". ACL(2021) [[PDF]](https://arxiv.org/pdf/2106.04489.pdf): leveraging hyper-network and task embedding for positive knowledge transfer between different tasks. Model architecture based on Houlsby Adapter.
* **Houlsby Adapter**: "Parameter-Efficient Transfer Learning for NLP". ICML(2019) [[PDF]](https://arxiv.org/abs/1902.00751): propose the method of freezing BERT parameters, only fine-tuning adapter layers, which generally preserves the BERT performance on GLUE.
* **AdapterFusion**: "AdapterFusion: Non-Destructive Task Composition for Transfer Learning". EACL(2021) [[PDF]](https://arxiv.org/pdf/2005.00247): fine-tuning Adapter layers on difference tasks. Then fuse the adapter layers together for better performance on a target task.
* **AdapterHub**: "AdapterHub: A Framework for Adapting Transformers". EMNLP(2020) [[PDF]](https://arxiv.org/abs/2007.07779)
* **MAD-X**: "MAD-X: An Adapter-Based Framework for Multi-Task Cross-Lingual Transfer". EMNLP(Pfeiffer et al., 2020): Multi-lingual transfer by adapters.

### Other Models
* **ExT5**: "ExT5: Towards Extreme Multi-Task Scaling for Transfer Learning". ICLR(2022) [[PDF]](https://arxiv.org/abs/2111.10952)
* **FLAN**: "Finetuned Language Models are Zero-Shot Learners". 2021 [[PDF]](https://arxiv.org/abs/2109.01652)
* **MUPPET**: "Muppet: Massive Multi-task Representations with Pre-Finetuning". EMNLP(2021) [[PDF]](https://aclanthology.org/2021.emnlp-main.468/)
* **T5**: "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer". JMLR(2019) [[PDF]](https://arxiv.org/abs/1910.10683): the text-to-text framework allows us to directly apply the same model, objective, training procedure, and decoding process to every task we consider.
* **MT-DNN**: "Multi-Task Deep Neural Networks for Natural Language Understanding". ACL(2019) [[PDF]](https://arxiv.org/abs/1901.11504): a setup at a scale of around 30 tasks and up to 440M parameters. 
* **PALs**: "BERT and PALs: Projected Attention Layers for Efficient Adaptation in Multi-Task Learning". ICML(2019) [[PDF]](http://proceedings.mlr.press/v97/stickland19a.html): MTL systems with shared BERT and task specific adapter layers, adapter layers include Houlsby Adapter and Parallel Attention Layers.
* **Task Hierachy**: "A Joint Many-Task Model: Growing a Neural Network for Multiple NLP Tasks". EMNLP(2017) [[PDF]](https://arxiv.org/abs/1611.01587): layers of tasks: POS -> Chunking -> Dependency parsing -> Semantic relatedness -> Entailment: input of each layer consists of label embeddings of all the previous layers and the hidden state of the previous layer.
### MTL Analysis
* **MetaWeighting**: "MetaWeighting: Learning to Weight Tasks in Multi-Task Learning" ACL Findings(2022) [[PDF]](https://aclanthology.org/2022.findings-acl.271/)
* **MTL v.s. IFT**: "When to Use Multi-Task Learning vs Intermediate Fine-Tuning for Pre-Trained Encoder Transfer Learning" ACL(2022) [[PDF]](https://arxiv.org/abs/2205.08124)
* **Task Gradients**: "Efficiently Identifying Task Groupings for Multi-Task Learning". NeurIPS spotlight(2021) [[PDF]](https://arxiv.org/abs/2109.04617)
* **Non-target Head**: "What's in Your Head? Emergent Behaviour in Multi-Task Transformer Models". EMNLP(2021) [[PDF]](https://arxiv.org/abs/2104.06129)
* **Ranking Transfer Languages**: "Ranking Transfer Languages with Pragmatically-Motivated Features for Multilingual Sentiment Analysis". EACL(2021) [[PDF]](https://arxiv.org/abs/2006.09336)
* **UnifiedQA**: "UNIFIEDQA: Crossing format boundaries with a single QA system". EMNLP-Findings(2020) [[PDF]](https://aclanthology.org/2020.findings-emnlp.171)
* **Task Embeddings**: "Exploring and predicting transferability
across NLP tasks". EMNLP(2020) [[PDF]](https://aclanthology.org/2020.emnlp-main.635)

### MTL Gradients
* **Gradient Vaccine**: "Investigating and Improving Multi-task Optimization in Massively Multilingual Models" ICLR(2021) [[PDF]](https://arxiv.org/abs/2010.05874)
* **Impartial MTL**: "Towards Impartial Multi-task Learning" ICLR(2021) [[PDF]](https://openreview.net/forum?id=IMPnRXEWpvr)
* **Pick a Sign**: "Just Pick a Sign: Optimizing Deep Multitask Models
with Gradient Sign Dropout" NeurIPS(2020) [[PDF]](https://arxiv.org/pdf/2010.06808.pdf)

## NLP in Finance
### Financial NLP Models
* **General FinBERT**: "FinBERT: A Pretrained Language Model for Financial Communications". 2020 [[PDF]](https://arxiv.org/abs/2006.08097)
* **FSA investigation**: "Financial Sentiment Analysis: An Investigation into Common Mistakes and Silver Bullets". CoLing(2020) [[PDF]](https://aclanthology.org/2020.coling-main.85.pdf): In the FSA domain, interpretability is essential. Also proposed common errors/difficulties in FSA, one numeracy related is about external information.
* **Sentiment FinBERT**: "FinBERT: Financial Sentiment Analysis with Pre-trained Language Models" 2019 [[PDF]](https://arxiv.org/abs/1908.10063)
### Financial NLP Tasks
* **FinQA**: "FinQA: A Dataset of Numerical Reasoning over Financial Data". EMNLP(2021) [[PDF]](https://arxiv.org/abs/2109.00122)
* **TAT-QA**: "TAT-QA: A Question Answering Benchmark on a Hybrid of Tabular and Textual Content in Finance". ACL(2021) [[PDF]](https://arxiv.org/abs/2105.07624)
* **NumClaim**: "NumClaim: Investor's Fine-grained Claim Detection". 2021 [[PDF]](http://mx.nthu.edu.tw/~chungchichen/papers/NumClaim.pdf)
* **NumeralAttachment**: "Overview of the NTCIR-15 FinNum-2 Task:
Numeral Attachment in Financial Tweets". 2020 [[PDF]](https://research.nii.ac.jp/ntcir/workshop/OnlineProceedings15/pdf/ntcir/01-NTCIR15-OV-FINNUM-ChenC.pdf)
* **FinCausal**: "Data Processing and Annotation Schemes for FinCausal Shared Task". 2020 [[PDF]](https://arxiv.org/abs/2012.02498)
* **Numeracy-600K**: "Numeracy-600K: Learning Numeracy for Detecting Exaggerated Information in Market Comments". ACL(2019) [[PDF]](https://aclanthology.org/P19-1635/)
* **NumberClassification**: "Overview of the NTCIR-14 FinNum Task: Fine-Grained Numeral Understanding in Financial Social Media Data". 2019 [[PDF]](https://www.cs.nccu.edu.tw/~hhhuang/docs/ntcir2019.pdf)
* **SemEval**: "SemEval-2017 Task 5: Fine-Grained Sentiment Analysis on Financial Microblogs and News". SemEval(2017) [[PDF]](https://aclanthology.org/S17-2089/)
* **TAP**: "Textual Analogy Parsing: What's Shared and What's Compared among Analogous Facts". EMNLP(2018) [[PDF]](https://arxiv.org/abs/1809.02700)
* **FiQA Sentiment Analysis and QA** [[Official Site]](https://sites.google.com/view/fiqa/)
* **Financial Phrase Bank**: "Good Debt or Bad Debt: Detecting Semantic Orientations in Economic Texts". 2013 [[PDF]](https://arxiv.org/abs/1307.5336)


## Numbers in NLP
### Number Representation
* **Overview**: "Representing Numbers in NLP: a Survey and a Vision". NAACL(2021) [[PDF]](https://arxiv.org/abs/2103.13136): Except a thorough survey about exisiting number representing technologies, this paper also provides a survey and taxonomy of tasks requiring numeracy.
* **Notation Comparison**: "Investigating the Limitations of Transformers with Simple Arithmetic Tasks". 2021 [[PDF]](https://arxiv.org/abs/2102.13019)
* **NumBERT**: "Do Language Embeddings capture Scales?". EMNLP-Findings(2020) [[PDF]](https://aclanthology.org/2020.findings-emnlp.439/): Scientific notation. Log-scale regression.
* **DICE**: "Methods for Numeracy-Preserving Word Embeddings". EMNLP(2020) [[PDF]](https://aclanthology.org/2020.emnlp-main.384/)
* **DigitRNN-sci & Exponent**: "An Empirical Investigation of Contextualized Number Prediction". EMNLP(2020) [[PDF]](https://aclanthology.org/2020.emnlp-main.385/)
* **Injecting Numeracy**: "Injecting Numerical Reasoning Skills into Language Models". ACL(2020) [[PDF]](https://aclanthology.org/2020.acl-main.89/): digit-level representation.
* **Numeracy Probing**: "Do NLP Models Know Numbers? Probing Numeracy in Embeddings". EMNLP(2019) [[PDF]](https://aclanthology.org/D19-1534/)
* **DigitCNN/RNN**: "Numeracy for Language Models: Evaluating and Improving their Ability to Predict Numbers" ACL(2018) [[PDF]](https://arxiv.org/abs/1805.08154)

### Numeracy
* **Reasoning Aware**: "Improving the Numerical Reasoning Skills of Pretrained Language Models" EMNLP(2022)[[PDF]](https://arxiv.org/pdf/2205.06733.pdf)
* **QDGAN**: "Question Directed Graph Attention Network for Numerical Reasoning over Text". EMNLP(2020) [[PDF]](https://arxiv.org/abs/2009.07448): numeray is highly related to the knowledge of number dependencies(e.g. what quantity is the number describing)
* **Injecting Numeracy**: "Injecting Numerical Reasoning Skills into Language Models". ACL(2020) [[PDF]](https://aclanthology.org/2020.acl-main.89/)
* **BERT Calculator**: "Giving BERT a Calculator: Finding Operations and Arguments with Reading Comprehension". EMNLP(2019) [[PDF]](https://aclanthology.org/D19-1609/)
* **Numeracy-600K**: "Numeracy-600K: Learning Numeracy for Detecting Exaggerated Information in Market Comments". ACL(2019) [[PDF]](https://aclanthology.org/P19-1635/)
* **DROP**: "DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs". NAACL(2019) [[PDF]](https://aclanthology.org/N19-1246/)

### Number in NLP Datasets
#### Arithmetic Word Problems
* **MathQA**: "MathQA: Towards Interpretable Math Word Problem Solving with Operation-Based Formalisms". NAACL(2019) [[PDF]](https://aclanthology.org/N19-1245/)
* **AquA**: "Program Induction by Rationale Generation : Learning to Solve and Explain Algebraic Word Problems". ACL(2017) [[PDF]](https://arxiv.org/abs/1705.04146)
#### Commonsense
* **Numbergame**: "Towards Question Format Independent Numerical Reasoning: A Set of Prerequisite Tasks". 2020 [[PDF]](https://arxiv.org/abs/2005.08516)
* **NumerSense**: "Birds have four legs?! NumerSense: Probing Numerical Commonsense Knowledge of Pre-Trained Language Models". EMNLP(2020)[[PDF]](https://aclanthology.org/2020.emnlp-main.557/)


## Relation Extraction
* **KGPool**: "KGPool: Dynamic Knowledge Graph Context Selection for Relation Extraction". ACL-Findings(2021) [[PDF]](https://arxiv.org/abs/2106.00459)
* **RECON**: "RECON: Relation Extraction using Knowledge Graph Context in a Graph Neural Network". WWW(2021) [[PDF]](https://arxiv.org/abs/2009.08694)
* **Joint Entity Relation Extraction**: "Effective Modeling of Encoder-Decoder Architecture for Joint Entity and Relation Extraction". AAAI(2020) [[PDF]](https://arxiv.org/abs/1911.09886)
* **R-BERT**: "Enriching Pre-trained Language Model with Entity Information for Relation Classification". 2019 [[PDF]](https://arxiv.org/abs/1905.08284)

## Semantic Role Labeling
* **Simple BERT**: "Simple BERT Models for Relation Extraction and Semantic Role Labeling". 2019 [[PDF]](https://arxiv.org/abs/1904.05255): provided baseline BERT model for relation extraction and semantic role labeling where the related entities and the predicates are given.

## Dependency Parsing
* **Trankit**: "Trankit: A Light-Weight Transformer-based Toolkit for Multilingual Natural Language Processing". EACL(2021) [[PDF]](https://aclanthology.org/2021.eacl-demos.10/)
* **STEPS**: "Applying Occam's Razor to Transformer-Based Dependency Parsing: What Works, What Doesn't, and What is Really Necessary". IWPT(2021) [[PDF]](https://aclanthology.org/2021.iwpt-1.13/)
* **UDify**: "75 Languages, 1 Model: Parsing Universal Dependencies Universally". EMNLP(2019) [[PDF]](https://aclanthology.org/D19-1279/)


## Domain Adaptation
* **Cross-domain Knowledge Distillation**: "Matching Distributions between Model and Data: Cross-domain Knowledge Distillation for Unsupervised Domain Adaptation". ACL(2021) [[PDF]](https://aclanthology.org/2021.acl-long.421.pdf)
* **UDALM**: "UDALM: Unsupervised Domain Adaptation through Language Modeling". NAACL(2019) [[PDF]](https://arxiv.org/abs/2104.07078): fine tuning on source domain labeled data while training a target domain MLM auxiliary task.
* **Adversarial BERT**: "Adversarial and Domain-Aware BERT for Cross-Domain Sentiment Analysis" ACL(2020) [[PDF]](https://aclanthology.org/2020.acl-main.370/): injecting target domain & domain knowledge through post-training, then apply domain-adversarial learning.
* **Domain Classification**: "Domain Adaptation with BERT-based Domain Classification and Data Selection". EMNLP(2019) [[PDF]](https://aclanthology.org/2020.acl-main.370/): use curriculum learning, defaults: only selects part of source domain data & need labeled target domain development set.


## Transformers Interpretation
* **Tutorial**: "Fine-grained Interpretation and Causation Analysis in Deep NLP Models". NAACL(2021) [[PDF]](https://arxiv.org/pdf/2105.08039.pdf)
* **BERT**: "What Does BERT Look At? An Analysis of BERT's Attention". ACL(2019) [[PDF]](https://aclanthology.org/W19-4828/):  An Analysis of BERT's Attention. BERT attention analysis [toolkit](https://github.com/clarkkev/attention-analysis)


## NLP in Programming Language
* **UniXcoder**: "Unified Cross-Modal Pre-training for Code Representation" ACL(2022) [[PDF]](https://arxiv.org/abs/2203.03850)
* **NeuralNetworkSolvesMath**: "A Neural Network Solves and Generates Mathematics Problems by Program Synthesis: Calculus, Differential Equations, Linear Algebra, and More". 2021 [[PDF]](https://arxiv.org/abs/2112.15594): Rephrasing the mathematics problems to programming tasks in a way that Codex can generate python code to solve them. An interesting application of PL language models.
* **CodeT5**: "CodeT5: Identifier-aware Unified Pre-trained Encoder-Decoder Models for Code Understanding and Generation". EMNLP(2021) [[PDF]](https://arxiv.org/abs/2109.00859): Another NL-PL based T5 model. Objectives: Masked Span prediction(auto-regressing) + Identifier prediction + Identifier Tagging + bimodal dual gnenration. MTL benefits code summarization and code refinement the most. The MTL system selects the best checkpoint for each task(like original T5).
* **SYNCoBERT**: "SynCoBERT: Syntax-Guided Multi-Modal Contrastive Pre-Training for Code Representation". 2021 [[PDF]](https://arxiv.org/abs/2108.04556): Triple modals: Natural Language, Programming Language, and Abstract Syntax Tree. Two new self-supervised objectives: identifier prediction and AST edge prediction, along with multi-modal contrastive training objective.
* **Codex**: "Evaluating Large Language Models Trained on Code". ACL(2021) [[PDF]](https://arxiv.org/abs/2107.03374): an NL-PL specific GPT. An interesting metric: Generate code samples and see whether the samples pass the unit tests.
* **TreeBERT**: "TreeBERT: A Tree-Based Pre-Trained Model for Programming Language". UAI(2021) [[PDF]](https://arxiv.org/abs/2105.12485): focusing on extracting syntactic and semantic information from AST
* **CoTexT**: "CoTexT: Multi-task Learning with Code-Text Transformer". ACL(2021) [[PDF]](https://arxiv.org/abs/2105.08645): an NL-PL domain specific version of T5, converting all tasks into a text-to-text format. Best Models are pretrained on C4 + CodeSearchNet(why best?). Multi-task: MT fine-tuning on 6 Programming languages.
* **PLBART**: "Unified Pre-training for Program Understanding and Generation". NAACL(2021) [[PDF]](Unified Pre-training for Program Understanding and Generation)
* **GraphCodeBERT**: "GraphCodeBERT: Pre-training Code Representations with Data Flow". ICLR(2021) [[PDF]](https://arxiv.org/abs/2009.08366): leverage the power of code structure(data flow). Some variables are not named after naming convention, therefore, data flow provides semantic information of the variables. Two new pre-training objectives: edge prediction and node alignment.
* **CodeXGLUE**: "CodeXGLUE: A Machine Learning Benchmark Dataset for Code Understanding and Generation". [[PDF]](https://arxiv.org/abs/2102.04664): contains a wide range of code-code, code-text, text-code, and text-text tasks. Three baselines: CodeBERT, CodeBERT + Decoder, CodeGPT.
* **CodeBERT**: "CodeBERT: A Pre-Trained Model for Programming and Natural Languages". EMNLP-Findings(2020) [[PDF]](https://arxiv.org/abs/2002.08155): treat code and natural language as multi-modal data. Pre-training CodeBERT using two objectives: Masked Language Modeling(MLM) and Replaced Token Detection(RTD).

## Data-to-text Generation
* **PlanGen**: "Plan-then-Generate: Controlled Data-to-Text Generation via Planning". EMNLP(2021) [[PDF]](https://arxiv.org/abs/2108.13740)

## Curriculum Learning
* **CL4NLU**: "Curriculum Learning for Natural Language Understanding". ACL(2020) [[PDF]](https://aclanthology.org/2020.acl-main.542/)