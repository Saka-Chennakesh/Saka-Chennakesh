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

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Saka Chennakesh",
        page_icon="👋",
        layout="wide"
    )

    st.write("# Saka Chennakesh Profile 👋")
     
#    st.sidebar.success("Select a demo above.")

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
    
with st.expander("See explanation"):
    st.markdown("""
    ### Technical Purview
        
    SAS (SAS Base, SAS Advanced, SAS SQL, SAS Macros, SAS Stat, SAS IML, SAS Graphs etc.,), SAS EM & SAS EG
    
    R Studio, R Open, Microsoft R, RevoscaleR, mrsdeploy
    
    Python (Spyder, ipython, Jupyter and other environments)-pandas, numpy, sklearn, statmodels, OpenCV, keras, tensorflow, pytorch, h2o, pycaret, CausalML, streamlit, Airflow, multiprocessing, Joblib, NLTK, Scipy, Spacy, pytorch transformers etc..)
    
    Other Tools/Environments: Hadoop (Hive, Pig, pyspark), ECL, Tableau , Alteryx and Power BI
    
    Credit Risk: Building and Validating Probability of Default, Loss Given Default, Exposure at Default, Credit Revenue Forecast, Interest Income and Non-Income forecast and many other models
    
    Market Risk: Building and Validating Value at Risk Models, Binomial Option pricing models, Black scholes models, Stochastic Models and GARCH, EWMA and other statistical, Mathematical and Financial Models for Banking sector.
    
    Consumer Analytics: Market Basket Analysis, Marketing Mixed Modelling, Media Mixed Modelling, Coverage Analysis, Segmentation, Clustering, Predictive Modelling, Sales forecasting, Churn Analysis, and Sampling Techniques.



    """
    )

    st.markdown(
        """
### Projects
#### Evoke Technologies:
1.	Invoice digitization to extract key invoice attributes from large variety of scanned/system generated invoices using Deep learning architectures such as Keras Retinanet, Yolov5, BERT for a Chemical Manufacturer.
2.	Extracting key attributes such as expiry date, manufacturer date, batches, chemical name, chemical composition from Scanned/System generated Certificate of Analysis for a Chemical Manufacturer using Deep learning architectures such as Keras Retinanet, Yolov5 and NLP based Transformer models.
3.	Checkbox detection/classification of different forms for one of the largest door manufacturers in North America using Deep learning architectures such as Keras Retinanet and Yolov5.
4.	Price Optimization using Statistical Models, Deep Learning (Yolov3 and MTCNN) to add loaded images for Online Retailer.
5.	Multilock Service Recommender using Statistical modelling, Brute Force Approach and Complex Statistical Customized Approach because the model is for Rare Event Prediction.
6.	Automated Extracting Attributes or content such as Registrant Name, State, Trading symbols, Financial Year, Risk Factors, Legal Proceedings, Safety Disclosures and Extracting Financial Statements from Form 10K, Form 8K and Form 10Q Annual/Quarterly Reports of any Company/Organization.  Used DL Models such as resnet, Yolo etc., and NLP models such as NER CRF, BERT etc.,
#### KPMG:
7.	PPNR-Prime Brokerage and Prime Financing model using time series regression, VARMAX, VECM, ARIMAX and other time series and quantitative methods
8.	Deposit Service charge models using time series regression and restricted VAR models
9.	Macro rate model using time series regression, linear equation solutions, spreads mechanism and dynamic back testing
10.	Ecuid predictions for Automobiles using multilevel chaid algorithm and ranger random forest algorithms
11.	PD model for petroleum engineering team using production and explored wells data
#### Prokarma:
12.	Variable rate irrigation model using mathematical business models
13.	Customized Sentiment Analysis using NLP, sentimentr, udpipe and several corpuses and lexicons
14.	Level predictions using word2vec, udpipe, lexicons and multilevel chaid algorithms
15.	Durable sales forecast model using boosting techniques, high performance Machine learning, Arimax, Varmax, Arch and Garch models
16.	Claim Process Automation by building the models for Claim Status Classification and Claim amount prediction using different ML models for one of the largest Fork uplifting Manufacturer
#### Genpact:
17.	Asset based lending models using time series regression, quantitate and qualitative methods
18.	Commercial and Auto loan models using logistic regression and other machine learning methods
19.	Auto leasing models using qualitative and quantitative statistical models to predict balances, balance run offs, Production spreads, revenues etc.,
20.	Employee cost to company model using dummy variable analysis, regression techniques, statistical assumptions, and quantitative methods.
21.	Propensity to Buy and Propensity to sell models
22.	Patient Enrolment Forecast, Drop out Prediction and Drop out Forecast for several Clinical Trails/Studies using Simulation Techniques, Several Distributions and Several Timeseries Models
#### TCS:
23.	Sales Value, Sales Volume and Penetration forecasts using Universe estimates, iterative proportional fitting, and different forecasting models for Australian Retailers.
24.	Campaign modelling analytics, trend analytics, supply chain analytics, consumer analytics for product level, SKU level management for Australian Retailers
25.	Retail loan Interest income forecast model using advanced mean based and variance-based forecasting models
26.	Retail Score card models using different types logistic regressions and assumption-based models
27.	Customer Attrition models using several machine learning and statistical models for Australian Retailers.

Above projects include extensive data extractions, data transformations, data analytics, model building, model validation, model implementation, model monitoring, model review, model deployment and/or product development.


    """
    )

    st.markdown(
        """
### Organisational Experience, Organisation Name, Designation
#### May’23 to till date	WELLS FARGO,	Bangalore	Vice President – Lead Model Solution Specialist
•	Leading and Developing Model Orchestrations tool for Banking Analytics, Strategies and Reporting by onboarding different types of Statistical, Mathematical and Interactive Models. Onboarding 1000+ models and creating interactive reporting for better strategies with accurate mechanism.

•	Part of Finance Transformation Office to architect, design and develop several business solutions
#### April’21 to May’23	Evoke Technologies,	Hyderabad		Data Science Manager
•	Used several Deep learning frameworks for Chemical Manufacture to Digitize their scanned Invoices and for Door Manufacturer to classify their different forms based on checkboxes.

•	Heading 10 members for Legal Domain DS & BI Team and working with 18 members Delivery and Practice teams

•	Closely coordinating with prospects, new/existing clients, Business development team to get new NLP, ML and DL Projects
#### Sept’19 to April’21	KPMG,	Bangalore		Assistant Manager (Modelling and Valuation)
•	Development of Statistical, Machine Learning Models as part of Development and Building Challenger or Benchmark Models as part of Validation.

•	Worked with CCAR/CECL Models, Campaign Models and Credit Risk Models.

•	Worked closely with Client-side Data Scientists, Petroleum engineers, Model Risk Management to Model as per their needs and/or to validate their models based on the requirements.

•	Interacted with Stake holders and Development Working Group to present the progress of Model Development

•	Mentored a team of 15 members and actively involving in talent search
#### Dec’17 to Sept’19 	Prokarma,	Hyderabad		Senior Technical Lead (DS/AI)
•	Created complex R packages for developing customized statistical, DS and AI models to meet business needs

•	Created Azure ML API’s, Stored Procedures that embedded with R codes to handle with large cloud-based data

•	Development of business model based on the Mathematical/Statistical and Data Science approach that interact with Web, Azure and IoT devices. 

•	Advanced modelling techniques that interact with Google maps, images, Json, prescriptions, Landsat and many other services that makes Complex Machines to work automatically based on environmental condition.

•	Handled with Big data using cores, parallel computation, memory management etc.,

•	Developed several sentiment analysis and customized text mining algorithms. 

•	Worked with Several ML models for Fork uplifting Truck manufacturer to automate their Warranty Claims and to predict the top 5 Truck Issues that can come across based on Truck related Features
#### Mar’15 to April’17 	GENPACT,	Bangalore	Assistant Manager (Designation)/Data Scientist for CSO Team
•	Manage a 6-member team and work with clients to assist them in credit risk and market risk projects pertaining to model development/validation, credit policy review, credit origination process review, and regulatory compliance, reporting

•	Hands on with Big Data techniques and worked on model development of PD, LGD, EAD, Stress Testing, Loss Forecasting, Credit Scoring, and other behavioural models. 

•	Developed complex financial and market risk models on Loss forecasting models, CCAR/DFAST Stress Testing, RWA & Capital Calculation, Credit Scoring (Retail Portfolios), Risk based Pricing, Credit VaR and Basel II/ICAAP models

•	Develop and applied machine learning and statistical analysis methods, such as classification, collaborative filtering, association rules, time-series analysis, advanced regression methods and hypothesis testing.
#### Jul’11 to Mar’15 	TATA Consultancy Services,	Bangalore		Senior Business Analyst
•	Suggest inputs and perform the steps in Model Building and Model Validation Strategies

•	Analyse and performing statistical data analysis on Credit & Market Risk, Retailer and Consumer verticals.

•	Built several Linear, Nonlinear and Time Series models and mentoring the team of size 10. 

•	Worked on Statistical modelling such as VIF, PCA, OLS, logistic regression, CART, CHAID and analytic/statistical analytics.


    """
    )

    st.markdown(
        """
#### Education
2016				Masters in Information Technology from Sikkim Manipal University (Distance)
2011				Masters in Statistics and Operation research from Hyderabad Central University
2009				B.Sc. in Statistics, Mathematics and Computer Science from Osmania University 
2006				12th from New Vision College, A.P Intermediate Education
2004				10th from Udaya Memorial High School, AP Education Board

#### Personal Details
Date of Birth:			February 12th, 1989

Present Address:	No: 7, PJR enclave road, near railway station, Lingampally- 500050


##### THANK YOU
##### Chennakesh S



    """
    )

if __name__ == "__main__":
    run()
