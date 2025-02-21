# InfoTech-Assistant-A-Multimodal-Conversational-Agent-for-InfoTechnology-Web-Portal-Queries
Natural Language Processing (NLP), Question Answering (QA) System, Large Language Models (LLMs), Retrieval Augmented Generation (RAG), Web Scraping, Infrastructure Technology.


---
abstract: |
  This pilot study presents the development of the InfoTech Assistant, a
  domain-specific, multimodal chatbot engineered to address queries in
  bridge evaluation and infrastructure technology. By integrating web
  data scraping, large language models (LLMs), and Retrieval-Augmented
  Generation (RAG), the InfoTech Assistant provides accurate and
  contextually relevant responses. Data, including textual descriptions
  and images of 41 bridge technologies, are sourced from publicly
  available documents on the InfoTechnology website and organized in
  JSON format to facilitate efficient querying. The architecture of the
  system includes an HTML-based interface and a Flask back end connected
  to the Llama 3.1 model via LLM Studio. Evaluation results show
  approximately 95 percent accuracy on domain-specific tasks, with high
  similarity scores confirming the quality of response matching. This
  RAG-enhanced setup enables the InfoTech Assistant to handle complex,
  multimodal queries, offering both textual and visual information in
  its responses. The InfoTech Assistant demonstrates strong potential as
  a dependable tool for infrastructure professionals, delivering high
  accuracy and relevance in its domain-specific outputs.
author:
   
- "InfoTech Assistant : A Multimodal Conversational Agent for
  InfoTechnology Web Portal Queries"
---

IEEEkeywords
Natural Language Processing (NLP), Question Answering (QA) System, Large
Language Models (LLMs), Retrieval Augmented Generation (RAG), Web
Scraping, Infrastructure Technology.
:::

# Introduction

The rapid growth of information in infrastructure technology has created
an increasing demand for advanced tools that streamline knowledge
retrieval and dissemination. With the continuous expansion of data in
fields such as engineering, construction, and maintenance, it has become
increasingly difficult for professionals to efficiently access and
utilize the vast amounts of knowledge available. The Federal Highway
Administration (FHWA) InfoTechnology [@b0] platform which acts as a
comprehensive repository, that consolidates a wide range of
infrastructure-related information, spanning domains such as bridges,
pavements, tunnels, and utilities. This platform is designed to support
decision-making processes and enhance the effectiveness of
infrastructure management by providing a centralized source of technical
data, standards, guidelines, and research findings

Our study specially focuses on bridge-related technologies, with an
emphasis on information pertaining to inspection, assessment, and
maintenance. While the abundance of structured information within the
platform is valuable, the sheer volume and complexity of content often
presents challenges for users to locate specific information within its
extensive content.

To address these challenges, the pilot *InfoTech Assistant* has been
developed as an interactive multimodal conversational tool, leveraging
Large Language Models (LLMs) and Natural Language Processing (NLP)
techniques to to enhance information retrieval and user interaction.
This system allows users to efficiently access relevant information
through a conversational interface. By employing a Retrieval-Augmented
Generation (RAG) approach [@b5], the InfoTech Assistant integrates
real-time data retrieval with language generation, ensuring precise,
contextually relevant responses tailored to bridge technology.

The InfoTech Assistant comprises several key components, including data
collection, a user-friendly interface [@b19], and integration with a
state-of-the-art LLM. Data collection employs automated web scraping
techniques to extract textual and visual data from the InfoTechnology
web portal [@b1]. The user interface allows users to submit queries and
receive detailed answers, supplemented with relevant images. Integrated
semantic retrieval capabilities ensure the assistant delivers precise,
domain-specific responses, making it highly effective for infrastructure
professionals and researchers.

The primary contributions are outlined as follows:

-   *Conversational Agent with Multimodal Integration:* The InfoTech
    Assistant enhances response accuracy by integrating both text and
    image data from the InfoTechnology platform, providing contextual
    understanding and visual support that caters to the needs of
    technical professionals in infrastructure domains[@b24].

-   *Domain-Specific Knowledge Retrieval Using RAG:* This system
    combines data retrieval with language generation capabilities,
    allowing for precise, adaptable responses optimized for complex,
    domain-specific queries in InfoTechnology.[@b5]

-   *User-Centric System Design for Real-Time Interaction:* The InfoTech
    Assistant's architecture, featuring a structured JSON database, a
    responsive HTML interface, and a Flask-based back end with LLM
    integration, ensures seamless, real-time access to information.
    [@b6; @b8]

-   *Quantitative Evaluation Framework for Response Quality:* Using
    cosine similarity and accuracy metrics, this evaluation framework
    rigorously assesses response relevance and precision, enhancing the
    reliability of the InfoTech Assistant in infrastructure-related
    queries.

-   *Domain-Specific Focus Tailored for Bridge Technology
    Professionals:* Designed specifically for professionals in bridge
    evaluation and infrastructure, the assistant delivers technical
    insights and comprehensive responses, supporting informed
    decision-making in the field.

This pilot study details the development of the InfoTech Assistant,
describing each component from data pre-processing to LLM integration
and deployment. It also elaborates on the RAG framework [@b21], which
combines reliable data retrieval with language generation to produce
accurate outputs. Finally, the paper discusses the application of the
InfoTech Assistant in real-world infrastructure settings, emphasizing
its role in providing quick and dependable access to critical
information.



# Related Work

The InfoTech Assistant draws inspiration from both general-purpose and
domain-specific Question Answering (QA) systems[@b15], leveraging
advancements in RAG[@b21], multi-modal integration, and NLP techniques.
By examining the strengths and limitations of these systems, the
InfoTech Assistant integrates proven approaches while addressing
challenges specific to infrastructure evaluation and Non-Destructive
Evaluation (NDE) technologies. The following review outlines key
contributions that informed the design and functionality of the InfoTech
Assistant.

## General-Purpose QA Systems

General-purpose QA systems, such as IBM Watson, have demonstrated robust
capabilities in natural language understanding and retrieval [@b9].
However, these systems often lack the domain-specific depth required for
technical fields like NDE. Frameworks such as SearchQA [@b9] and DrQA
[@b10] introduced techniques for semantic similarity and TF-IDF-based
query alignment, which, while effective for open-domain QA, exhibited
limited accuracy in handling specialized content. The proposed InfoTech
Assistant builds on these approaches by utilizing semantic embeddings
for precise query matching, thereby enhancing retrieval accuracy in
domain-specific contexts.

## Domain-Specific QA Systems

BioASQ [@b11] highlights the successful application of Named Entity
Recognition (NER) and domain-specific NLP for handling specialized
corpora. This demonstrates the need for tailored approaches in technical
domains. Similarly, Haystack [@b12] integrates RAG frameworks to deliver
highly accurate and contextually grounded responses. These methodologies
influenced the design of our InfoTech assistant by emphasizing the
importance of combining semantic retrieval with generative modeling to
address complex infrastructure-related queries effectively.

## Chatbot Applications in Construction

Applications like Skanska's Sidekick [@b13] showcase the utility of
chatbots in construction, providing natural language interfaces to
access construction data. However, these tools focus on logistical
queries and lack depth for technical evaluations. Our InfoTech Assistant
extends these capabilities by incorporating detailed material-specific
insights, making it suitable for NDE scenarios[@b35] where precise and
contextual responses are critical.

## Multi-Modal and RAG Integration

Systems such as MS MARCO [@b14] and Visual QA [@b15] demonstrate the
value of multi-modal data in enriching user interactions by integrating
structured data with image-text retrieval. The InfoTech Assistant adopts
similar methodologies by structuring its dataset in JSON format,
enabling efficient retrieval of both textual and visual information.
Inspired by large language models like Turing-NLG [@b16], the Assistant
leverages Llama 3.1 [@b7] to ensure scalability, secure query handling,
and advanced generative capabilities.

# Methodology

The development of the InfoTech Assistant follows a multi-phase
methodology, encompassing data collection via web scraping [@b20],
front-end interface development, and back-end integration with a
pre-trained large language model (LLM) to enable efficient information
retrieval and user interaction.

## The System Architecture

The architecture comprises several essential components, each fulfilling
a vital role in enabling seamless data processing, precise information
retrieval, and responsive user interaction, as illustrated in Fig.
which was created using Lucidchart 

::: figure*
![image](https://github.com/user-attachments/assets/36a1b373-643c-4561-8ad0-73864883c562)

:::

### Connector

The Connector component bridges the front-end interface and back-end
systems, ensuring seamless communication and real-time interaction. In
this implementation, Flask acts as the back-end intermediary,
transmitting user queries to the LLM via POST requests and retrieving
generated responses. Flask formats the responses, incorporating both
text and images, before delivering them back to the interface for user
display. This architecture ensures efficient query processing and
enhances system responsiveness.

### Database and Storage

This component manages structured storage for pre-processed data,
including text and images. Organized in JSON format, this structure
ensures quick and consistent retrieval for system queries.

### Process Manager

The Process Manager component oversees key tasks such as data
pre-processing, keyword extraction, semantic matching, and response
generation. It ensures the alignment of retrieved data with user
requests.

### Display

The Display component provides the user interface for query input and
response visualization, ensuring an intuitive and functional user
experience.

## Data Collection and Pre-Processing

The data collection process is implemented via an automated web scraping
pipeline using Selenium [@b18]. This pipeline specifically targets 41
bridge-related technologies under the Bridge section on the
InfoTechnology website [@b1], systematically extracting textual
descriptions and corresponding images. The scraped data is organized
into a structured JSON format, enabling efficient access and streamlined
processing for InfoTech Assistant operations. Sample data for the
technologies \"Hammer Sounding\" and \"Magnetic Particle Testing (MT)\"
[@b1] are illustrated in TABLE 
respectively. This structured format allows for rapid retrieval and
contextual presentation of information, supporting the system's ability
to provide accurate and relevant responses based on user queries.

To enhance the dataset, additional infrastructure-related domains,
including pavements, tunnels, and utilities, were identified for
inclusion [@b0]. Publicly available datasets were scraped using Selenium
[@b18] and preprocessed using a consistent pipeline aligned with the
bridge data. The datasets were structured in JSON format, integrating
textual and visual data to enable efficient querying. This expansion
broadens the dataset's scope, equipping the InfoTech Assistant to handle
diverse infrastructure-related queries effectively.

## Transformer Model

The all-mpnet-base-v2 model , developed by Microsoft, is a
transformer-based architecture designed for semantic similarity tasks
and integral to the InfoTech Assistant's language understanding
capabilities . Trained on over 1 billion sentence pairs, this
compact 420 MB model combines efficient memory usage with robust
performance across diverse domains. Its key features include mean
pooling to summarize semantic content and normalized embeddings to
improve reliability and accuracy in similarity scoring. These features
make it well-suited for enabling the InfoTech Assistant to provide
precise and contextually relevant responses.

## Flask Integration

Flask is a lightweight and flexible web framework [@b8] that facilitates
real-time communication between the front end and back end in the
InfoTech Assistant system. It routes user queries via HTTP requests to
the model's processing components, ensuring efficient response handling.
Flask's scalability allows the system to maintain robust performance
even under increasing loads. By integrating Flask with
Retrieval-Augmented Generation (RAG) , the application provides
accurate, context-rich, and timely responses, making it a reliable
choice for conversational AI implementations.

## LLM Model Integration

### The Llama 3.1 Model

Llama 3.1, an advanced decoder-only transformer model [@b25], forms the
core of the InfoTech Assistant, enabling it to handle complex,
context-aware queries. Its architecture incorporates cutting-edge
training techniques such as supervised fine-tuning (SFT) [@b32] and
direct preference optimization (DPO), which enhance data quality and
response precision. The quantization from BF16 to FP8 ensures
computational efficiency, enabling deployment on single-server nodes
. The model's capabilities, including a 128K context window and
multi-turn conversation support, make it ideal for tasks requiring
continuity and detailed, dynamic responses.

### The Mistral Model

The Mistral Model  a lightweight and efficient transformer-based
language model, also contributes to the InfoTech Assistant, enabling it
to handle complex, context-aware queries. Its design emphasizes
low-latency performance while maintaining high-quality responses. The
compact architecture of Mistral allows for efficient deployment and
processing, which is particularly advantageous for real-time
applications.

### Temperature Parameters in Language Models

A language model is characterized by numerous hyperparameters that
govern its architecture, training process, and performance. Temperature
is one such hyperparameter that modulates randomness in model outputs by
scaling the logits before applying the softmax function [@b26]. The
Temperature parameter $T$ affects the probability distribution as
follows[@b31]:

![image](https://github.com/user-attachments/assets/0132d413-1993-404e-a420-3b5dc5627c16)


where $z_i$ represents the logits, $T$ is the temperature, and $P(x_i)$
is the probability of token $i$ from Equation
[\[eq: temp\]](#eq: temp){reference-type="ref" reference="eq: temp"}.

A low temperature (e.g., 0.1) produces deterministic outputs by
emphasizing high-probability tokens, resulting in consistent but less
diverse responses. A high temperature (e.g., 1.5) flattens the
probability distribution, promoting greater variability and creativity
in responses but at the cost of predictability.

The InfoTech Assistant employs a temperature setting of 0.7 to balance
deterministic accuracy with conversational diversity [@b27]. This
setting enables contextually relevant and adaptable responses, ensuring
precise information retrieval for technical queries while maintaining a
natural and engaging conversational tone.

## Evaluation Metrics

### Cosine Similarity

In this \"InfoTech Assistant\" system, cosine similarity is used to
evaluate the relevance of retrieved content to user queries, ensuring
that responses closely match the semantic intent of the input[@b4].

Cosine similarity is a mathematical measure used to determine the
similarity between two non-zero vectors in an $n$-dimensional space
[@b28]. It is widely used in natural language processing to compare the
semantic similarity of vectorized text representations. The formula for
cosine similarity is as follows:

![image](https://github.com/user-attachments/assets/d5437831-31ad-476a-aaa4-5068a8679d35)


From Equation [\[eq: cos\]](#eq: cos){reference-type="ref"
reference="eq: cos"} where $\vec{A}$ and $\vec{B}$ are the vectors being
compared, $\vec{A} \cdot \vec{B}$ is their dot product, and
$\|\vec{A}\|$ and $\|\vec{B}\|$ are their magnitudes.

Cosine similarity values range from $-1$ to $1$. A value of $1$
indicates identical vectors, $0$ indicates orthogonality (no
similarity), and $-1$ indicates complete dissimilarity.

Cosine similarity is scale-invariant, meaning it focuses on the
orientation of the vectors rather than their magnitude, which makes it
particularly suitable for comparing normalized text embeddings.

### Response Accuracy

Accuracy, in this study, is calculated based on a threshold applied to
cosine similarity scores[@b4]. To determine accuracy, a predefined
threshold (e.g., 0.85) is applied to cosine similarity scores. If the
similarity score for a response meets or exceeds this threshold, the
response is considered "correct". Accuracy is computed as the ratio of
correct responses to the total number of test cases, represented as a
percentage as shown in Equation


![image](https://github.com/user-attachments/assets/a225a5ec-40dd-441e-b7b0-4fd96846168e)


# Experimental Results and Analysis

The pilot InfoTech Assistant integrates key components, including data
collection, a user-friendly HTML interface [@b6], and a state-of-the-art
LLM. Automated web scraping extracts textual and visual data on 41
bridge technologies from the InfoTechnology web portal [@b1], organizing
it into a structured JSON format for efficient querying. The intuitive
interface enables users to submit queries and receive detailed,
image-enhanced responses, with Flask facilitating seamless communication
between the front end and the LLM.
![image](https://github.com/user-attachments/assets/595914c2-e0e6-4976-bc20-e0d89836c305)

The performance of the InfoTech Assistant [@b34] was evaluated based on
response accuracy, latency, and user satisfaction. The system underwent
testing with both technical and non-technical users, and its performance
metrics were analyzed over multiple testing rounds.

![InfoTechnology Web Portal UI with InfoTech Assistant
Interaction]

![image](https://github.com/user-attachments/assets/66cc6baf-1b1f-44f7-a440-14a29004c277)



## Experiment Setup and System Requirements

For the experiment, the Assistant system was deployed using an LM Studio
server  hosting the Llama 3.1 8B model[@b25], which was accessed
through a locally run API. Once the LM Studio server is initiated, a
front-end HTML page containing the InfoTech Assistant interface is
launched. Users interact with the InfoTech Assistant by inputting
queries through this interface. The server, running the large language
model, processes these queries [@b24], applies NLP techniques, and
generates responses that are displayed in the chat interface as shown in

Additionally, if the context contains relevant images, they are
displayed alongside the text.

## Sample Responses and Image Retrieval Results

 illustrates sample interactions with the InfoTech
Assistant, showcasing its capability to produce accurate, contextually
relevant responses and retrieve related images. The InfoTech Assistant
leverages RAG[@b7] to ensure fact-based answers are derived from the
structured dataset, providing validated responses rather than generating
content independently. Additionally, the system retrieves images
relevant to user queries, thereby enriching the responses with visual
context. For complex inquiries, the LLM further enhances user
understanding by generating concise summaries[@b22]. Multi-image
retrieval supports comprehensive insight, especially for technical
topics requiring a visual aid for clarity.

## Comparison of Bot and LLM Responses

In the InfoTech Assistant system, responses from the \"Bot\" and the
\"LLM\" serve distinct but complementary roles. The Bot response is
directly generated from data scraped from the official InfoTechnology
website, ensuring the integrity and precision of the information
provided. This approach allows the Bot response to present a
comprehensive and detailed answer that closely reflects the technical
content from the original source, thereby maintaining factual accuracy
essential for professional use as shown in Fig.


In contrast, the LLM response is a summarized version of the same
information, crafted to enhance user comprehension and readability. By
distilling the primary points, the LLM response provides an accessible
summary that allows users to quickly capture essential insights[@b22].
This dual-response mechanism effectively addresses diverse user needs by
combining detailed, source-based responses with a more concise,
user-friendly summary as shown in Fig.


## Evaluation and Analysis

The performance of the InfoTech Assistant was evaluated using key
metrics, focusing on response accuracy and contextual relevance. These
metrics were derived from the Assistant's ability to accurately retrieve
content and deliver appropriate responses, including visual data when
available.

### Contextual Relevance

: The InfoTech Assistant effectively managed contextually relevant
queries, but exhibited limitations when addressing highly specific or
nuanced queries [@b24]. For instance, ambiguous questions occasionally
returned broadly related information instead of precise answers [@b19].
This highlights potential areas for improvement, such as enhanced
fine-tuning or the adoption of a larger model [@b28], to better handle
complex contextual inquiries.



### Latency and Scalability

The latency of the InfoTech Assistant, defined as the response time from
receiving a user query to delivering an answer, was measured during
testing. The Llama 3.1 model exhibited a latency range of 15 to 20
seconds, primarily due to local processing demands. In comparison, the
Mistral-7B-Instruct-v0.2 model demonstrated a latency range of 10 to 22
seconds, reflecting its efficiency in query processing. Latency was
observed to be dependent on the system's computational capabilities,
with increased processing power reducing response times [@b34]. The test
system configuration is detailed in TABLE


### Similarity Calculation and Accuracy Evaluation

The accuracy of the InfoTech Assistant was evaluated using cosine
similarity, a widely used metric in natural language processing to
measure semantic alignment between vector embeddings of text [@b29].
Cosine similarity values, calculated using Equation
range from 0 to 1, with scores of 0.85 or
higher considered correct.

Expected and actual responses were vectorized using the
Sentence-Transformer model [@b2], and accuracy was determined as the
percentage of correct responses among the total test cases. As shown in
 the Llama 3.1 model achieved
similarity scores of 0.94 and 0.92, and accuracies of 95% and 94% for
queries like \"What is Electrical Resistivity\" and \"What are benefits
of Hammer Sounding,\" respectively. Similarly, as presented in TABLE
for the same queries the Mistral 7B
model achieved similarity scores of 0.90 and 0.92, with accuracies of
92% and 94%. These results demonstrate the Assistant's capability to
provide semantically aligned and contextually accurate responses.

![image](https://github.com/user-attachments/assets/b2ec07b2-f5b6-4a33-9aba-74176349dbe4)

  : Sample Questions Similarity and Accuracy Results for Mistral 7B




The comparison between the pre-trained models Llama 3.1 and Mistral 7B
was conducted over 15 rounds of testing, using 15 different questions
related to various technologies available on the InfoTechnology Bridge
website. The similarity scores and overall accuracies for both models
were calculated and analyzed.

![image](https://github.com/user-attachments/assets/e3db38c2-66c9-41df-b4be-4b43d1d2f6ea)


illustrates the individual similarity scores for each sample, with
overall accuracies represented by horizontal dashed lines. The Llama 3.1
model [@b25] achieves an overall accuracy of 95%, which is marginally
superior to the Mistral 7B model [], with an accuracy of 93%. These
visualizations effectively highlight the consistency and performance
differences between the models, with Llama 3.1 8B demonstrating enhanced
semantic alignment in comparison to Mistral 7B.

 provides a comprehensive breakdown of
similarity and accuracy scores for each question across both models.
This detailed representation offers valuable insights into the granular
performance of Llama 3.1 8B and Mistral 7B on specific tasks,
complementing the observations presented in Fig.


# Conclusions

The InfoTech Assistant demonstrates the potential of conversational
agents in addressing domain-specific challenges in infrastructure
technology. By employing advanced techniques such as data scraping,
Retrieval-Augmented Generation (RAG), and a large language model (LLM)
hosted on LLM Studio, the system delivers accurate and contextually
relevant information. Its architecture, comprising structured data
collection, a Flask-based backend, and a user-friendly interface,
enables efficient and precise responses to complex queries related to
bridge technologies.

The integration of RAG significantly enhances the accuracy of LLM
responses by grounding them in factual and domain-specific data,
establishing the InfoTech Assistant as a reliable resource for
infrastructure professionals. Evaluations validate its capability to
manage diverse queries, offering both textual and visual outputs.
Additionally, the use of pre-trained LLMs ensures versatility, allowing
the system to provide detailed answers from a locally stored JSON
database while leveraging a broader knowledge base for general queries.

In terms of performance, the Llama 3.1 8B model achieved an accuracy of
95%, outperforming the Mistral 7B model, which recorded an accuracy of
93%. However, the Mistral 7B model exhibited lower latency, highlighting
a trade-off between accuracy and response time. Interestingly, the
similarity metric showed limitations in fully evaluating the models'
capabilities. For example, in certain cases, the Llama model provided
correct and detailed answers but achieved a lower similarity score due
to its tendency to include additional explanatory content. This finding
suggests that while similarity can be indicative, it may not be the most
reliable metric for assessing model performance in this context.

# Future Work

Future enhancements for the InfoTech Assistant will prioritize reducing
latency and improving its capability to manage multi-turn conversations
effectively. Key advancements include the integration of advanced models
such as Falcon 180B b36], which are designed to deliver low-latency
and contextually rich responses, alongside improvements to
Retrieval-Augmented Generation (RAG) for more accurate and precise
content retrieval. Additionally, the system will leverage
domain-specific large language models (LLMs) and fine-tuning techniques
[@b28] to enhance its performance and relevance in addressing technical
queries. To support these advancements, the system will adopt
cloud-based infrastructure to enable faster data access and scalable
operations. A dynamic re-scraping mechanism will also be implemented to
ensure that responses remain aligned with the most up-to-date
information. These enhancements aim to significantly improve the
system's contextual awareness, responsiveness, and overall user
experience, solidifying its role as a reliable tool for infrastructure
technology professionals.

# Acknowledgment {#acknowledgment .unnumbered}

The authors would like to thank Meta and Mistral, the team behind the
LLaMA model, for providing a powerful language tool that was
instrumental in the development of the InfoTech Assistant. The authors
also extend their gratitude to the contributors of the Transformers
library, whose tools facilitated the seamless integration of the system.
Additionally, the authors acknowledge the support of the Federal Highway
Administration (FHWA) for granting access to the publicly available
InfoTechnology web portal, the data from which was crucial for the
training and testing of the model. These contributions were vital in
making this project successful and effective.

::: thebibliography
00

Federal Highway Administration. \"FHWA InfoTechnology Platform:
Supporting Decision-Making for Infrastructure Management\". U.S.
Department of Transportation. 2020.

H. S. Park and B. D. Smith. \"Centralized Information Systems in
Infrastructure Management: A Case Study on the FHWA InfoTech Platform\".
International Journal of Civil Engineering, 2017, 15(6), 495-503.

Federal Highway Administration. "FHWA InfoTechnology," Dot.gov.
https://infotechnology.fhwa.dot.gov/ (accessed Nov. 15, 2024).

FHWA InfoTechnology. "Bridge FHWA InfoTechnology," Dot.gov.
<https://infotechnology.fhwa.dot.gov/bridge/>. (Accessed: Nov. 15,
2024).

Xiao, Yunze. (2022). A Transformer-based Attention Flow Model for
Intelligent Question and Answering Chatbot. 167-170.
10.1109/ICCRD54409.2022.9730454.(accessed: Nov. 12, 2024).

LM Studio. "LM Studio - Experiment with Local LLMs."
<https://lmstudio.ai/>.

P. P. Ghadekar, S. Mohite, O. More, P. Patil, Sayantika, and S.
Mangrule, \"Sentence Meaning Similarity Detector Using FAISS,\" *2023
7th International Conference On Computing, Communication, Control And
Automation (ICCUBEA)*, Pune, India, 2023, pp. 1--6. doi:
[10.1109/ICCUBEA58933.2023.10392009](10.1109/ICCUBEA58933.2023.10392009){.uri}.(accessed:
Nov. 15, 2024).

Alfredodeza. "GitHub - alfredodeza/learn-retrieval-augmented-generation:
Examples and demos on how to use Retrieval Augmented Generation with
Large Language Models."
GitHub.<https://github.com/alfredodeza/learn-retrieval-augmented-generation>.(accessed:
Nov. 01, 2024).

Anghelescu, P., & Nicolaescu, S. V. (2018, June). Chatbot application
using search engines and teaching methods. In 2018 10th international
conference on electronics, computers and artificial intelligence (ECAI)
(pp. 1-6). IEEE.

Cabezas, D. S., Fonseca-Delgado, R., Reyes-Chacón, I., Vizcaino-Imacana,
P., & Morocho-Cayamcela, M. E. Integrating a LLaMa-based Chatbot with
Augmented Retrieval Generation as a Complementary Educational Tool for
High School and College Students.(accessed: Nov. 01, 2024).

Singh, V., Rohith, Y., Prakash, B., & Kumari, U. (2023, May). ChatBot
using Python Flask. In 2023 7th International Conference on Intelligent
Computing and Control Systems (ICICCS) (pp. 1182-1185). IEEE.(accessed:
Nov. 01, 2024).

Dunn, M., Sagun, L., Higgins, M., Guney, V. U., Cirik, V., & Cho, K.
(2017). Searchqa: A new q&a dataset augmented with context from a search
engine. arXiv preprint arXiv:1704.05179.(accessed: Nov. 15, 2024.)

Li, Y., Li, W., & Nie, L. (2022). Dynamic graph reasoning for
conversational open-domain question answering. ACM Transactions on
Information Systems (TOIS), 40(4), 1-24.

Dimitriadis, D. (2023). Machine learning and natural language processing
techniques for question answering (Doctoral dissertation, ARISTOTLE
UNIVERSITY OF THESSALONIKI).

John J. "Jack" McGowan, \"Chapter 14 Project Haystack Data Standards,\"
in Energy and Analytics BIG DATA and Building Technology Integration ,
River Publishers, 2015, pp.237-243.

M. Thibault, "Meet Sidekick, Skanska's new AI chatbot," Construction
Dive, Mar. 06, 2024. \[Online\]. Available:
https://www.constructiondive.com/news/sidekick-skanskas-ai-construction-chatbot/709350/.(Accessed:
Nov. 10, 2024)

Bajaj, P., Campos, D., Craswell, N., Deng, L., Gao, J., Liu, X., \... &
Wang, T. (2016). Ms marco: A human generated machine reading
comprehension dataset. arXiv preprint arXiv:1611.09268.

Antol, S., Agrawal, A., Lu, J., Mitchell, M., Batra, D., Zitnick, C. L.,
& Parikh, D. (2015). Vqa: Visual question answering. In Proceedings of
the IEEE international conference on computer vision (pp. 2425-2433).

H. K. Skrodelis, A. Romanovs, N. Zenina and H. Gorskis, \"The Latest in
Natural Language Generation: Trends, Tools and Applications in
Industry,\" 2023 IEEE 10th Jubilee Workshop on Advances in Information,
Electronic and Electrical Engineering (AIEEE), Vilnius, Lithuania, 2023,
pp. 1-5, doi: 10.1109/AIEEE58915.2023.10134841.

LucidChart. "Diagramming Powered by Intelligence." Available:
<https://www.lucidchart.com/>.(accessed: Nov. 01, 2024)

Stack Overflow. \"Using Selenium to click page and scrape info from
routed page.\" Available:
<https://stackoverflow.com/questions/71786531/using-selenium-to-click-page-and-scrape-info-from-routed-page>.
Accessed: Nov. 02, 2024.

Thosani, P., Sinkar, M., Vaghasiya, J., & Shankarmani, R. (2020, May). A
self learning chat-bot from user interactions and preferences. In 2020
4th International Conference on Intelligent Computing and Control
Systems (ICICCS) (pp. 224-229). IEEE.
[10.1109/ICICCS48265.2020.9120912](10.1109/ICICCS48265.2020.9120912){.uri}.

M. G, M. M, S. R and I. Ritharson P, \"Chatbots Embracing Artificial
Intelligence Solutions to Assist Institutions in Improving Student
Interactions,\" 2023 International Conference on Circuit Power and
Computing Technologies (ICCPCT), Kollam, India, 2023, pp. 912-916, doi:
10.1109/ICCPCT58313.2023.10245835.

Omrani, P., Hosseini, A., Hooshanfar, K., Ebrahimian, Z., Toosi, R., &
Akhaee, M. A. (2024, April). Hybrid Retrieval-Augmented Generation
Approach for LLMs Query Response Enhancement. In 2024 10th International
Conference on Web Research (ICWR) (pp. 22-26). IEEE. doi:
[10.1109/ICWR61162.2024.10533345](10.1109/ICWR61162.2024.10533345){.uri}.

Zeng, Jicheng; Liu, Xiaochen; and Fang, Yulin, \"Influence of
Leaderboard and Trial Space on Large Language Models Popularity\"
(2024). ICIS 2024 Proceedings. 9.
https://aisel.aisnet.org/icis2024/digital_emergsoc/digital_emergsoc/9

Jiang, A. Q., Sablayrolles, A., Mensch, A., Bamford, C., Chaplot, D. S.,
Casas, D. D. L., \... & Sayed, W. E. (2023). Mistral 7B. arXiv preprint
arXiv:2310.06825.

Ait-Mlouk, A., & Jiang, L. (2020). KBot: a Knowledge graph based chatBot
for natural language understanding over linked data. IEEE Access, 8,
149220-149230. doi:
[10.1109/ACCESS.2020.3011848](10.1109/ACCESS.2020.3011848){.uri}.

He, Z., Shu, W., Ge, X., et al. (2024). Llama Scope: Extracting Millions
of Features from Llama-3.1-8B with Sparse Autoencoders. arXiv preprint
arXiv:2410.20526.

Bengio, Y., Goodfellow, I., & Courville, A. (2017). Deep learning (Vol.
1). Cambridge, MA, USA: MIT press.(Accessed: Nov. 14, 2024).

Perković, G., Drobnjak, A., & Botički, I. (2024, May). Hallucinations in
llms: Understanding and addressing challenges. In 2024 47th MIPRO ICT
and Electronics Convention (MIPRO) (pp. 2084-2088). IEEE. doi:
[10.1109/MIPRO60963.2024.10569238](10.1109/MIPRO60963.2024.10569238){.uri}.

Rosati, R., Antonini, F., Muralikrishna, N., Tonetto, F., & Mancini, A.
(2024, September). Improving Industrial Question Answering Chatbots with
Domain-Specific LLMs Fine-Tuning. In 2024 20th IEEE/ASME International
Conference on Mechatronic and Embedded Systems and Applications (MESA)
(pp. 1-7). IEEE. doi:
[10.1109/MESA61532.2024.10704843](10.1109/MESA61532.2024.10704843){.uri}.

S. Bhattacharjee, A. Das, U. Bhattacharya, S. K. Parui and S. Roy,
\"Sentiment analysis using cosine similarity measure,\" 2015 IEEE 2nd
International Conference on Recent Trends in Information Systems
(ReTIS), Kolkata, India, 2015, pp. 27-32, doi:
10.1109/ReTIS.2015.7232847.

Jayanthi, S. M., Embar, V., & Raghunathan, K. (2021). Evaluating
pretrained transformer models for entity linking in task-oriented
dialog. arXiv preprint arXiv:2112.08327.

Peeperkorn, M., Kouwenhoven, T., Brown, D., & Jordanous, A. (2024). Is
temperature the creativity parameter of large language models?. arXiv
preprint arXiv:2405.00492.

Dong, G., Yuan, H., Lu, K., Li, C., Xue, M., Liu, D., \... & Zhou, J.
(2023). How abilities in large language models are affected by
supervised fine-tuning data composition. arXiv preprint
arXiv:2310.05492.

Perez, S. P., Zhang, Y., Briggs, J., Blake, C., Levy-Kramer, J.,
Balanca, P., \... & Fitzgibbon, A. W. (2023). Training and inference of
large language models using 8-bit floating point. arXiv preprint
arXiv:2309.17224.

D'Urso, S., Martini, B., & Sciarrone, F. (2024, July). A Novel LLM
Architecture for Intelligent System Configuration. In 2024 28th
International Conference Information Visualisation (IV) (pp. 326-331).
IEEE.

Vrana, J., Meyendorf, N., Ida, N., & Singh, R. (2022). Introduction to
NDE 4.0. In: Meyendorf, N., Ida, N., Singh, R., & Vrana, J. (eds)
Handbook of Nondestructive Evaluation 4.0. Springer, Cham. Retrieved
from <https://doi.org/10.1007/978-3-030-73206-6_43>

Almazrouei, E., Alobeidli, H., Alshamsi, A., Cappelli, A., Cojocaru, R.,
Debbah, M., \... & Penedo, G. (2023). The falcon series of open language
models. arXiv preprint arXiv:2311.16867.
:::

Supplementary Data Table: Scraped Data for Post ID 2769 - Hammer Sounding Technology
![image](https://github.com/user-attachments/assets/d6f0540e-73ae-45dd-9d26-ab0d828e09d0)

Supplementary Data Table: Scraped Data for Post ID 129 - Magnetic Particle Testing (MT) Technology

![image](https://github.com/user-attachments/assets/292464ba-74e7-433f-a294-4fd97db23c1b)

Supplementary Data Table: Scraped Data for Post ID 129 - Magnetic Particle Testing (MT) Technology

![image](https://github.com/user-attachments/assets/9ade4fac-6f51-49bf-b6bb-34fb662968fe)

SimilarityandAccuracyScoresofSampleQuestionsforLlamaandMistralModels

![image](https://github.com/user-attachments/assets/da8473cd-b411-4fea-89ba-5d7d0eb94432)



