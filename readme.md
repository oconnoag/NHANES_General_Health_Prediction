# Introduction

Assessing subjective feelings (e.g. general health, quality of life, etc) has been an important task in medicine for as long as the profession has existed.  Considering that a core goal in medicine seeks to restore positive subjective perceptions and/or prevent negative subjective perceptions in the subjects, the salience of accurate measurements is not hard to overstate.  As with any subjective assessment, however, a number of biases can introduce a large amount of arbitrary variance to the measurements.  There have been several attempts to standardize measurements of subjective experience using various survey questions--one example is called the Nottingham Health Profile, an attempt to measure "quality of life" (Hunt, 1985).  To expand to the existing list of subjective-to-objective mapping, we are attempting to predict a particular subjective measurement ("General Health") from CDC health data by building statistical models with laboratory-based data, sociological-based data, and combined subsets of the two.  The general hypothesis for this project is given a certain set of information about a patient, we can determine/predict how they feel about their health (by classifying them into a corresponding category), without explicitly asking them.

## Data Description

#### NHANES Overview
The data used in this study is compiled and propogated by the CDC under the program name National Health and Nutrition Examination Survey (which will herein be referred to as NHANES).  The NHANES program operates annually (though with different pools of participants) with the goal of assessing health and nutrition of the denizens of America.  The data produced by the program is unique because it utilizes biochemical laboratory-based methods, questionnaires, diet, external body measures, and demographics.  This abundance of data is attractive for statisticians and data scientists (or students of the fields, like us) for generating models for all sorts of different outcomes using a number of measurements (referred to as features in data science).

#### Data acquistion
The specific NHANES data used herein is a subset of files and fields from the 2015-2016 collection (https://wwwn.cdc.gov/nchs/nhanes/continuousnhanes/default.aspx?BeginYear=2015).  This is the most recent release of NHANES data, as it takes years for the researchers to organize, clean, and verify the data from a particular year.
As noted previously, the NHANES data is broken up into several categories (e.g. demographics, laboratory, etc), and each category is subsequently broken up into a number of files which are made up a number of columns corresponding to the particular measurement.  Interestingly, the data files within each category are released to the public in XPT format, which cannot readily be used for analysis in Pandas.  To be able to use these datafiles in our study, we wrote a shell script that coverts the xpt files to csv.  

We have placed the converted files in a github repository here: https://github.com/oconnoag/NHANES_Data.  This repository also contains the compiled files, which will be described shortly.  We opted for having separate repositories for the data and the analyses, because (1) if anyone else would like the download the data themselves in csv format, they can easily clone or zip the repository and (2) this architecture allows for easy reading of the csv files from Github, so there is no need for someone to download the entire datasets to their local machine to view our analysis.  

#### Selecting intial features to compile for consideration
For the compiled data (i.e. the data that we actually analyzed for incorporation in our models), we decided to choose a subset of fators that we surmised would correlate with a subject's general health.  In the data setup files (found here:  https://github.com/oconnoag/NHANES_Data/tree/master/compiled_data), the features we divided between the authors of this study (Alijah and Flor).  Alijah would choose and explore data from the laboratory category, and Flor would choose and explore data from the questionnaire and demographics categories.  By including data from these two disparate fields we get to utilize both quantitative and qualitative features.  What's more, this approach serves as an initial test to target specific factors out of intuition (sort of a proof of concept); however, once these original models are constructed, we may switch gears and generalize our approach by including many more factors.

#### General Health
The general_health field is originally coded as the "HSD010" column in the /Questionnaire/HSQ_I ("General Health Status") file.  The measurements come from subjects answering the question:  "Would you say {your/SP's} health in general is {List Options}?".  The answers are broken up into 5 levels: 1, 2, 3, 4, 5 corresponding to "Excellent", "Very Good", "Good", "Fair", and "Poor", respectively.  These categories were further mapped to 3 classes, due to poor model performance (likely caused by extreme class inbalances).  The mapping was {classes 1-2: 1, class 3: 2, classes 4-5: 3} and were described as {1: "Better", 2: "Good", 3: "Worse"}.

#### Laboratory Features Considered
| Filename_NHANES | Filename_Project | Feature_Name_NHANES | Feature_Name_Project | Description |
|------|------|------|------|------|
|   BIOPRO_I  | standard_biochem_profile | LBXSBU | Blood Urea Nitrogen  | Measured from blood in mg/dL |
|   BIOPRO_I  | standard_biochem_profile | LBXSC3SI | Bicarbonate  | Measured from serum in mmol/L |
|   BIOPRO_I  | standard_biochem_profile | LBXSCA | Total Calcium | Measured from serum in mg/dL
|   BIOPRO_I  | standard_biochem_profile | LBXSCH | Cholesterol | Measured from serum in mg/dL
|   BIOPRO_I  | standard_biochem_profile | LBXSCLSI | Chloride | Measured from serum in mmol/L
|   BIOPRO_I  | standard_biochem_profile | LBXSGL | Glucose | Measured from serum in mg/dL
|   BIOPRO_I  | standard_biochem_profile | LBXSIR | Iron | Measured from serum in ug/dL
|   BIOPRO_I  | standard_biochem_profile  | LBXSKSI | Potassium | Measured from serum in mmol/L
|   BIOPRO_I  | standard_biochem_profile  | LBXSNASI | Sodium | Measured from serum in mmol/L
|   BIOPRO_I  | standard_biochem_profile  | LBXSTP | Total Protein | Measured from serum in g/dL
|   BIOPRO_I  | standard_biochem_profile  | LBXSTR | Triglycerides | Measured from serum in mg/dL
|   BIOPRO_I  | standard_biochem_profile  | LBXSUA | Uric acid | Measured from serum in mg/dL
|   TST_I  | sex_steroid_hormone | 	LBXTST | Testosterone | Measured from serum in ng/dL
|   TST_I  | sex_steroid_hormone | 	LBXEST | Estradiol | Measured from serum in pg/dL
|   TST_I  | sex_steroid_hormone | LBXSHBG | Sex Hormone Binding Globulin (SHBG) | Measured from serum in nmol/L
|   GHB_I  | glycohemoglobin | LBXGH | glycohemoglobin | Measured from serum as a percentage (%)


#### Questionnaire/Demographic Features

| Filename_NHANES | Filename_Project | Feature_Name_NHANES | Feature_Name_Project | Description |
|------|------|------|------|------|
|   DEMO_I  | demographics | RIDAGEYR | Age  | Age in years |
|   DEMO_I  | demographics | DMDHHSIZ | People in Household  | Values range 1-7 |
|   DEMO_I  | demographics | INDHHIN2 | Annual household income | Values range 1-15, representing incomes 0 to $100,000+
|   DBQ_I  | diet_and_behavior | DBQ700 | Healthy Diet | Values range from 1-5
|   DBQ_I  | diet_and_behavior | DBD900 | Number of fast food meals past 7 days | Values range from 0 to 21+
|   DBQ_I  | diet_and_behavior | DBD910 | Number of frozen meals past 30 days | Values range from 0 to 90+
|   PAQ_I  | physical_activity | PAQ635 | Walk or bike | Answers are Y/N
|   PAQ_I  | physical_activity  | PAQ650 | Vigorous recreational activities per week | Answers are Y/N
|   PAQ_I  | physical_activity  | PAQ680 | Minutes of sedentary activity per day | Values range from 0 to 1380
|   BMX_I  | body_measures  | BMXBMI | Body Mass Index | Measured in kg/m**2
