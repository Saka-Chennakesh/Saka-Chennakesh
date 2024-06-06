# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
from streamlit_option_menu import option_menu
import pandas as pd

LOGGER = get_logger(__name__)



def run():
    st.set_page_config(
        page_title="Saka Chennakesh",
        page_icon="👋",
        layout="wide"
    )

    st.write("# Saka Chennakesh 👋")
    st.image('photo.jpg', caption='Profile Photo', width=200)
#    st.sidebar.success("Select a demo above.")
    #"Profile", "Skills",  "Projects", 'Org'
    def on_change(key):
        selection = st.session_state[key]
        st.write(f"Selection changed to {selection}")
    
    selected3 = option_menu(None, ["Summary", "Skills",  "Projects", 'ORG',"Education"],
                            icons=['list-task', 'list-task', "list-task", 'gear','list-task'],
                            on_change=on_change, key='menu_5', orientation="horizontal")
    selected3        
    if selected3=="Summary":
        st.markdown(
            """
    ### Profile Summary
    ##### Resume Headline:
    Data Science, Engineering and Analytics Manager/Architect with 12 years of experience in Machine Learning, Financial Modelling, Statistical Modelling, NLP, Deep Learning, MLOps, Data Engineering and Data Analytics using Python, R, SAS, pyspark, git, docker, AWS, Azure, Tableau, Alteryx, Web Frameworks, SQL, NoSQL, Excel etc.,
    Currently working at WELLS FARGO, Bangalore as Vice President -Lead Quant Model Solutions Specialist and PhD Scholar in Deep Learning at Capital University, Jharkhand.

    ##### Business Domains: 
    Worked/Lead with deep technical, data science, strategic and analytical skills for: [Retail, Manufacturer, Banking (Credit Risk, Market Risk), Insurance, Irrigation, Pharma and Legal] Domain Clients 

    ##### Modelling Strategies: 
    Hands on with Model Development, Model Validation, Model Deployment, Model Monitoring, A/B Testing, Model Risk Management, Model Governance, Model Review & Model Documentation.

    ##### Modelling Techniques: 
    •	Hands on with Machine Learning Models (Decision Trees, Random Forests, Boosting Models (XGB, GBM, Adaboost), Ensemble Models, SVM, KNN, Apriori Algorithm, Brute Force Modelling, Stacking and Blending Models), Statistical Modelling, Clustering, Quantitative Modelling, Optimization and Simulation Techniques.

    •	Worked with Time Series Models such as ARIMAX, VARMAX, GARCH, Prophet, LSTM etc.,

    •	Hands on with Deep Learning Models (Perceptron, RNN, CNN, BiLSTM, BERT, RoBERTa etc.,) and Architectures (YOLO, resent, densenet, vggnet, effnet models) using tools such as Keras, TensorFlow, Theano, Torch, Caffe, bert, pytorch

    •	Developed and Automated image processing, NLP, NLU, NLG, word/doc embeddings, Word2vec, LDA, LSA, transformer-based models using packages such as SKLEARN, mlr, caret, H2OAutoML, CausalML, MXNet, NLTK, CNTK, MLlib, pyspark

    ##### Financial Risk Modelling:
    •	Hands on with PD, LGD, EAD, ICAAP, CCAR, PPNR and DFAST Stress testing, Basel Norms and other credit and market risk frame works.

    •	Hands on with Validation process that includes model validation/ Risk Management guide lines such as SR 11/7 and SS 3/18, model development document, testing, and benchmarking

    •	Knowledge of various market data instruments such as equities, IR curves, options, volatilities etc.

    •	Worked on Model Developments, Validations, Monitoring, Annual Reviews & Implementations with Various LOBs

    ##### Data Engineering, MLops & AIOps: 
    •	Experienced with MLOps – scalable development to deployment of complex data science workflows in AWS Sage maker, Azure, Snowpark, Alteryx Gallery and using Python flask, Dash, Streamlit Web Interface, Alteryx, APIs, Airflow, CI/CD Pipelines to create AI based accelerators/Solutions.

    •	Worked with platforms such as Snowflake, dremio, FDLZ and with several databases (Azure Cosmos DB, Retail Factory-SQL, MS-SQL, MySQL, Greenplum, HAWQ, MongoDB).

    •	Ability to use cloud services such as AWS Textract/form recognizer, Azure ML 

    ##### Analytics & Business Intelligence:
    •	Develop visual reports, Data Analytics, Feature Engineering, dashboards and KPI scorecards

    •	Extensive experience in building analytical solutions such as Pricing and promotional effectiveness, Customer segmentation, Customer LTV optimization, Cross Sell, Upsell, Market basket Analysis, Media Mixed Optimization

    •	Perform Data Preparation, Treatment, Data Audit Report, data assessment, univariate distribution analysis, logical checks.

    •	Experience in managing analytics projects from inception to delivery, that includes development and Automation

    •	Perform deep dive analysis of key business trends from multiple perspectives and package the insights

    ##### Management: 
    •	Leading multiple delivery squads (consisting of Analysts, Scientists and BI Engineers) to help deliver on key projects that includes prototyping, designing, and requirement analysis.

    •	Responsible for developing solutions to solve complex business problems with large structured/unstructured data and identify new opportunities to improve productivity

    •	Closely coordinating with prospects, new/existing clients, Business development team to get new Projects

    •	Lead multiple teams to perform various complex activities in analytics, development, maintenance of CECL, Basel and CCAR models for Home Lending, Auto and Unsecured (Cards, Loans, Business Loans,) portfolios.

    •	Coordinating with various stakeholders in line of business, model implementation team, Model Risk Management, Model Governance to ensure flawless and timely delivery of models in compliance with Model Risk Policies and Regulatory Frameworks.

        """
        )

    if selected3=="Skills":
        dataset = pd.read_csv("skills.csv",index_col = False)
        st.markdown(dataset.style.hide(axis="index").to_html(), unsafe_allow_html=True)
        
    if selected3=="Projects":

    ### Projects
    #### Evoke Technologies:
        with st.expander("Evoke Technologies"):
            st.markdown("""
        1.	[Invoice digitization](https://www.evoketechnologies.com/blog/invoice-digitization-using-resnet50-yolov5-and-adobe-ocr/) to extract key invoice attributes from large variety of scanned/system generated invoices using Deep learning architectures such as Keras Retinanet, Yolov5, BERT for a Chemical Manufacturer.
        2.	Extracting key attributes such as expiry date, manufacturer date, batches, chemical name, chemical composition from Scanned/System generated Certificate of Analysis for a Chemical Manufacturer using Deep learning architectures such as Keras Retinanet, Yolov5 and NLP based Transformer models.
        3.	[Checkbox detection/classification](https://www.evoketechnologies.com/blog/yolo-modeling-for-checkboxes-detection-classification/) of different forms for one of the largest door manufacturers in North America using Deep learning architectures such as Keras Retinanet and Yolov5.
        4.	[Price Optimization](https://www.evoketechnologies.com/blog/how-to-use-deep-ai-for-product-image-influence-on-ecommerce-sales-and-prices/) using Statistical Models, Deep Learning (Yolov3 and MTCNN) to add loaded images for Online Retailer.
        5.	Multilock Service Recommender using Statistical modelling, Brute Force Approach and Complex Statistical Customized Approach because the model is for Rare Event Prediction.
        6. [Evoke Data Studio](https://www.evoketechnologies.com//wp-content/uploads/2023/09/Data-Studio.mp4) for Data Preparation, Data Cleansing, Data Transformation, Data Summarization, Data Visualization etc., Evoke TrueVision introduces a new level of automated image processing, seamlessly integrating advanced capabilities for tasks such as Image Classification, which labels entire images to simplify categorization; Object Detection, identifying and locating multiple objects; and Image Segmentation.
        7. [Evoke True ML](https://www.evoketechnologies.com//wp-content/uploads/2023/09/TrueML.mp4) for an automated machine learning service that can tackle a wide array of data science challenges, to solve complex problems in Classification, Regression, Clustering, Forecasting and Anomaly Detection. TrueML’s intelligent model selection algorithm ensures that each dataset is matched with the most suitable machine learning model, optimizing for performance and accuracy.
        8. [Evoke True Vision](https://www.evoketechnologies.com/wp-content/uploads/2024/03/Truevision_cropped.mp4) provides extensive customization, enabling users to choose from various deep learning architectures such as ResNet, ImageNet, VGGNet, etc.
        9.	Automated Extracting Attributes or content such as Registrant Name, State, Trading symbols, Financial Year, Risk Factors, Legal Proceedings, Safety Disclosures and Extracting Financial Statements from Form 10K, Form 8K and Form 10Q Annual/Quarterly Reports of any Company/Organization.  Used DL Models such as resnet, Yolo etc., and NLP models such as NER CRF, BERT etc.,
            """)
        
    #### KPMG:
        with st.expander("KPMG"):
            st.markdown("""
    7.	PPNR-Prime Brokerage and Prime Financing model using time series regression, VARMAX, VECM, ARIMAX and other time series and quantitative methods
    8.	[Deposit Service charge models](https://github.com/Saka-Chennakesh/Saka-Chennakesh/blob/main/Projects) using time series regression and restricted VAR models
    9.	Macro rate model using time series regression, linear equation solutions, spreads mechanism and dynamic back testing
    10.	Ecuid predictions for Automobiles using multilevel chaid algorithm and ranger random forest algorithms
    11.	PD model for petroleum engineering team using production and explored wells data
        """)
    #### Prokarma:
        with st.expander("Prokarma"):
            st.markdown("""
    12.	Variable rate irrigation model using mathematical business models
    13.	Customized Sentiment Analysis using NLP, sentimentr, udpipe and several corpuses and lexicons
    14.	Level predictions using word2vec, udpipe, lexicons and multilevel chaid algorithms
    15.	Durable sales forecast model using boosting techniques, high performance Machine learning, Arimax, Varmax, Arch and Garch models
    16.	Claim Process Automation by building the models for Claim Status Classification and Claim amount prediction using different ML models for one of the largest Fork uplifting Manufacturer
            """)
        
    #### Genpact:
        with st.expander("GENPACT"):
            st.markdown("""
    17.	Asset based lending models using time series regression, quantitate and qualitative methods
    18.	Commercial and Auto loan models using logistic regression and other machine learning methods
    19.	Auto leasing models using qualitative and quantitative statistical models to predict balances, balance run offs, Production spreads, revenues etc.,
    20.	Employee cost to company model using dummy variable analysis, regression techniques, statistical assumptions, and quantitative methods.
    21.	Propensity to Buy and Propensity to sell models
    22.	Patient Enrolment Forecast, Drop out Prediction and Drop out Forecast for several Clinical Trails/Studies using Simulation Techniques, Several Distributions and Several Timeseries Models
            """)
    #### TCS:
        with st.expander("TCS"):
            st.markdown("""
    23.	Sales Value, Sales Volume and Penetration forecasts using Universe estimates, iterative proportional fitting, and different forecasting models for Australian Retailers.
    24.	Campaign modelling analytics, trend analytics, supply chain analytics, consumer analytics for product level, SKU level management for Australian Retailers
    25.	Retail loan Interest income forecast model using advanced mean based and variance-based forecasting models
    26.	Retail Score card models using different types logistic regressions and assumption-based models
    27.	Customer Attrition models using several machine learning and statistical models for Australian Retailers.
            """)

        st.markdown("""
    Above projects include extensive data extractions, data transformations, data analytics, model building, model validation, model implementation, model monitoring, model review, model deployment and/or product development.
        """)

    if selected3=="ORG":
         # with st.expander("Evoke Technologies"):
            
            dataset = pd.read_csv("org.csv", encoding='ISO-8859-1',index_col = False)
            #st.markdown(dataset.to_html(escape=False), unsafe_allow_html=True)
            st.markdown(dataset.iloc[0:1,0:4].style.hide(axis="index").to_html(), unsafe_allow_html=True)
            st.markdown(dataset.iloc[0:1,4:6].style.hide(axis="index").to_html(), unsafe_allow_html=True)
            
    if selected3=="Education":

        dataset = pd.read_csv("edu.csv",index_col = False)
        st.markdown(dataset.style.hide(axis="index").to_html(), unsafe_allow_html=True)

if __name__ == "__main__":
    run()
