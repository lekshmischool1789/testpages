---
title: Data Integration and Cleaning
nav_include: 1
---
# Part 1 - Data Integration & Cleaning

## Load Census Data



```python
import pandas as pd
import urllib
import numpy as np
import warnings
```




```python
"""
Function
---------------
split_MSA

This method takes in a dataframe with MSA and splits into a city_key (largest city)
and state_key. This will help facilitate MSA merging

Returns dataframe with these two additional features
"""
def split_MSA(df):
    df['MSA'] = df['MSA'].str.replace('Metro Area', '')
    # Need to manually fix how this MSA is written
    df.loc[df['MSA'].str.contains("Texarkana"), "MSA"] = "Texarkana, AR-TX"

    #Grab Everything before comma
    df['city_key'] = df['MSA'].str.split(",").str[0]
    # Then grab everything before first hyphen if it has it
    df['city_key'] = df['city_key'].str.split("-").str[0].str.strip()
    # State will be everying after comma 
    df['state_key']=df['MSA'].str.split(",").str[1].str.strip()
    return(df)

"""
Function
--------
append_df

This function appends two dataframes

Parameters:
    input - dataframe to be appended
    output - dataframe to be appended onto
    
Returns a single dataframe 
"""
def append_df(input,output):
    if output.empty:
        output=input.copy()
    else:
        output=pd.concat([output,input])
        output.reset_index(drop='Index',inplace=True)
    return(output)

'''
Function
-----------
var_thresh

This function takes in a dataframe and keeps only those varaibles that have a pct
non-missing that is above that threshold
'''
def var_thresh(df, thresh=0.65):
    return(df.loc[:, pd.notnull(df).sum() > len (df) *thresh])

'''
Function
---------
slim_df

This function takes in a list of variables to keep
on the the given df. It keep the variables + geography
then renames to MSA and drops the first row of variable descriptions
'''
def slim_df(df, var_list):
    var_list.append('GEO.display-label')
    df = df.loc[:, var_list]
    # Get rid of Micro Areas
    df = df.loc[~df['GEO.display-label'].str.contains("Micro Area"), :]
    
    df = df.rename(index=str, columns={'GEO.display-label': 'MSA'})
    df['MSA'] = df["MSA"].astype(str)
    # Drop first row of var descriptions
    df = df.loc[df.MSA != "Geography", :]
    # Split MSA into city-state key
    return(split_MSA(df))

'''
Function
---------
match_crime

This function will take in a dataframe and make changes to MSA
in order to match crime data
'''
def match_crime(df):
    df.loc[df['MSA'].str.contains('Crestview'),'city_key']='Crestview'
    df.loc[df['MSA'].str.contains('Sarasota'),'city_key']='North Port'
    df.loc[df['MSA'].str.contains('Louisville'),'city_key']='Louisville'
    df.loc[df['MSA'].str.contains('Santa Maria'),'city_key']='Santa Maria'
    df.loc[df['MSA'].str.contains('Weirton'),'city_key']='Weirton'
    df.loc[df['MSA'].str.contains('San Germán'),'city_key']='San German'
    df.loc[df['MSA'].str.contains('Mayagüez'),'city_key']='Mayaguez'
    df.loc[df['MSA'].str.contains('Honolulu'),'city_key']='Urban Honolulu'

    #State
    df.loc[df['MSA'].str.contains('Worcester'),'state_key']='MA-CT'
    df.loc[df['MSA'].str.contains('Myrtle Beach'),'state_key']='SC-NC'
    df.loc[df['MSA'].str.contains('Salisbury'),'state_key']='MD-DE'
    df.loc[df['MSA'].str.contains('Weirton'),'state_key']='WV-OH'
    return(df)

'''
Function
--------
get_file_name

Get the appropriate file name giving year and table code

'''
def get_file_name(year, table_code):
    if year == 2006:
        mid = 'EST'
    else:
        mid = '1YR'
    return('ACS_'+str(year)[2:]+"_%s_" %mid + table_code)

'''
Function
--------
convert_to_int

This function takes in a dataframe and list of vars to convert to int
'''
def convert_to_int(df, int_vars):
    df[int_vars] = df[int_vars].astype(int)
    return(df)
'''
Function
----------
create_proportions

This function will take in a list of variables and a single total variable
It then creates proportions by dividing each of the variables in the list by the total
to create a proportion
'''
def create_proportions(df,num_list, total_var):
    df.loc[:, num_list] = df[num_list].apply(lambda x: x / df[total_var])
    del df[total_var]
    return(df)

"""
function
-----------
fbi_url_generator

This function pulls violent crime spreadsheets from FBI UCR website
for a given year

It takes in the year of interest and outputs a url string
"""
def fbi_url_generator(year):
    if 2006 <= year <= 2009:
        return('https://www2.fbi.gov/ucr/cius%i/data/documents/'%year +str(year)[2:]+'tbl06.xls')
    else:
        if 2010 <= year <= 2011:
            end = '/tables/table-6/output.xls'
        elif 2012 <= year <= 2013:
            end = '/tables/6tabledatadecpdf/table-6/output.xls'
        elif 2014 <= year <= 2015:
            if year == 2014:
                mid = 'Table_6_Crime_in_the_United_States_by_Metropolitan_Statistical_Area_2014/output.xls'
            else:
                mid = 'table_6_crime_in_the_united_states_by_metropolitan_statistical_area_%i.xls/output.xls' %year
            end = '/tables/table-6/%s' %mid
        elif year == 2016:
            end ='/tables/table-4/table-4/output.xls' 
        hostname = 'https://ucr.fbi.gov/crime-in-the-u.s/%i/crime-in-the-u.s.-%i' %(year, year)
        return(hostname + end)
    
```




```python
#####################
# Employment Data
#####################
emp_all = pd.DataFrame()
for year in range(2006, 2017):
    f = get_file_name(2006, 'S2301')
    employ = pd.read_csv("data/employ/%s.csv" %f, encoding='Latin-1')
    
    # Grab Unemployment
    un = [v for v in employ.columns if "HC04" in v and "EST" in v]
    employ = slim_df(employ, un)
    
    employ = employ.loc[:, ["MSA", "city_key", "state_key", 
                          "HC04_EST_VC01", "HC04_EST_VC03",
                         'HC04_EST_VC24']]
    employ['year'] = year
    emp_all = append_df(employ, emp_all) 

# Process Final DataFrame
emp_all = emp_all.sort_values(['city_key', 'state_key', 'year'])
emp_all = match_crime(emp_all)
del emp_all['MSA']
emp_all = emp_all.rename(index=str,
                        columns={'HC04_EST_VC01': 'unemp_16_ovr',
                                'HC04_EST_VC03': 'unemp_16_19',
                                'HC04_EST_VC24': 'unemp_female'})
emp_all.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city_key</th>
      <th>state_key</th>
      <th>unemp_16_ovr</th>
      <th>unemp_16_19</th>
      <th>unemp_female</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abilene</td>
      <td>TX</td>
      <td>6.6</td>
      <td>23.1</td>
      <td>5.2</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>367</th>
      <td>Abilene</td>
      <td>TX</td>
      <td>6.6</td>
      <td>23.1</td>
      <td>5.2</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>734</th>
      <td>Abilene</td>
      <td>TX</td>
      <td>6.6</td>
      <td>23.1</td>
      <td>5.2</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1101</th>
      <td>Abilene</td>
      <td>TX</td>
      <td>6.6</td>
      <td>23.1</td>
      <td>5.2</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>1468</th>
      <td>Abilene</td>
      <td>TX</td>
      <td>6.6</td>
      <td>23.1</td>
      <td>5.2</td>
      <td>2010</td>
    </tr>
  </tbody>
</table>
</div>





```python
############
# Age Data
############
age_all = pd.DataFrame()
for year in range(2006, 2017):
    f = get_file_name(year, 'S0101')
    age = pd.read_csv("data/age/%s.csv" %f, encoding='Latin-1')
    age = slim_df(age, [v for v in age.columns if "EST" in v])
    age = age.replace("(X)", np.nan)

    age = age.loc[:, ['MSA','city_key','state_key',
                      'HC01_EST_VC33','HC01_EST_VC34',
                      'HC01_EST_VC01', 'HC02_EST_VC01',
                      'HC03_EST_VC01', 'HC01_EST_VC06',
                      'HC01_EST_VC07', 'HC02_EST_VC07']]
    age['year'] = year
    age_all = append_df(age, age_all) 


# Process Final DataFrame
age_all = age_all.sort_values(['city_key', 'state_key', 'year'])
age_all = age_all.rename(index=str,
                         columns={'HC01_EST_VC33':'median_age',
                                'HC01_EST_VC34': 'sex_ratio',
                                'HC01_EST_VC01': 'total_pop',
                                'HC02_EST_VC01': 'male_pop',
                                'HC03_EST_VC01': 'female_pop',
                                'HC01_EST_VC06': 'pop_15_19',
                                'HC01_EST_VC07': 'pop_20_24',
                                'HC02_EST_VC07': 'male_pop_20_24'})

# Convert to Int and Get Proportions
age_all = convert_to_int(age_all, ['total_pop', 'male_pop', 'female_pop'])
age_all = create_proportions(age_all, ['male_pop', 'female_pop'], 'total_pop')
# Match Crime Data and then get rid of MSA
age_all = match_crime(age_all)
del age_all['MSA']
age_all.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city_key</th>
      <th>state_key</th>
      <th>median_age</th>
      <th>sex_ratio</th>
      <th>male_pop</th>
      <th>female_pop</th>
      <th>pop_15_19</th>
      <th>pop_20_24</th>
      <th>male_pop_20_24</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abilene</td>
      <td>TX</td>
      <td>34.4</td>
      <td>99.1</td>
      <td>0.497717</td>
      <td>0.502283</td>
      <td>8.3</td>
      <td>8.7</td>
      <td>10.2</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>367</th>
      <td>Abilene</td>
      <td>TX</td>
      <td>34.9</td>
      <td>99.1</td>
      <td>0.497777</td>
      <td>0.502223</td>
      <td>9.5</td>
      <td>7.7</td>
      <td>8.6</td>
      <td>2007</td>
    </tr>
    <tr>
      <th>736</th>
      <td>Abilene</td>
      <td>TX</td>
      <td>34.6</td>
      <td>101.0</td>
      <td>0.502381</td>
      <td>0.497619</td>
      <td>9.2</td>
      <td>7.6</td>
      <td>8.9</td>
      <td>2008</td>
    </tr>
    <tr>
      <th>1105</th>
      <td>Abilene</td>
      <td>TX</td>
      <td>33.2</td>
      <td>97.0</td>
      <td>0.492269</td>
      <td>0.507731</td>
      <td>7.9</td>
      <td>9.0</td>
      <td>9.6</td>
      <td>2009</td>
    </tr>
    <tr>
      <th>1479</th>
      <td>Abilene</td>
      <td>TX</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.501355</td>
      <td>0.498645</td>
      <td>7.3</td>
      <td>9.5</td>
      <td>9.9</td>
      <td>2010</td>
    </tr>
  </tbody>
</table>
</div>





```python
###############
# Income Data
###############
inc_all = pd.DataFrame()
for year in range(2006, 2017):
    f = get_file_name(year, 'B19001F')
    inc = pd.read_csv("data/house_income/%s.csv" %f, encoding='Latin-1')
    # Keep only the estimates
    inc = slim_df(inc, [v for v in inc.columns if "HD01" in v])
    inc['year'] = year
    inc_all = append_df(inc, inc_all) 

# Proccess Final Data Frame
inc_all =  inc_all.rename(index=str,
                          columns={'HD01_VD01':'total',
                                  'HD01_VD02': 'inc_lt10',
                                  'HD01_VD03': 'inc_10_15',
                                  'HD01_VD04': 'inc_15_19',
                                  'HD01_VD05': 'inc_20_24',
                                  'HD01_VD06': 'inc_25_29',
                                  'HD01_VD07': 'inc_30_34',
                                  'HD01_VD08': 'inc_35_39',
                                  'HD01_VD09': 'inc_40_44',
                                  'HD01_VD10': 'inc_45_49',
                                  'HD01_VD11': 'inc_50_59',
                                  'HD01_VD12': 'inc_60_74',
                                  'HD01_VD13':'inc_75_99',
                                  'HD01_VD14':'inc_100_124',
                                  'HD01_VD15':'inc_125_149',
                                  'HD01_VD16':'inc_150_199',
                                  'HD01_VD17':'inc_gt_200'})

numeric_vars =  [v for v in inc_all.columns if "inc" in v]
inc_all = convert_to_int(inc_all, numeric_vars)
inc_all['total'] = inc_all['total'].astype(int)
# Get propotion of each imcome bracket by dividing by total
inc_all = create_proportions(inc_all, numeric_vars, "total")
# Match Crime data and Get rid of MSA
inc_all = match_crime(inc_all)
del inc_all['MSA']
inc_all.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>inc_lt10</th>
      <th>inc_10_15</th>
      <th>inc_15_19</th>
      <th>inc_20_24</th>
      <th>inc_25_29</th>
      <th>inc_30_34</th>
      <th>inc_35_39</th>
      <th>inc_40_44</th>
      <th>inc_45_49</th>
      <th>inc_50_59</th>
      <th>inc_60_74</th>
      <th>inc_75_99</th>
      <th>inc_100_124</th>
      <th>inc_125_149</th>
      <th>inc_150_199</th>
      <th>inc_gt_200</th>
      <th>city_key</th>
      <th>state_key</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.066006</td>
      <td>0.039355</td>
      <td>0.191044</td>
      <td>0.142857</td>
      <td>0.034552</td>
      <td>0.051751</td>
      <td>0.094980</td>
      <td>0.044159</td>
      <td>0.004803</td>
      <td>0.138829</td>
      <td>0.060738</td>
      <td>0.099628</td>
      <td>0.004029</td>
      <td>0.006508</td>
      <td>0.018748</td>
      <td>0.002014</td>
      <td>Abilene</td>
      <td>TX</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.042883</td>
      <td>0.102766</td>
      <td>0.078153</td>
      <td>0.191829</td>
      <td>0.056077</td>
      <td>0.063436</td>
      <td>0.136514</td>
      <td>0.011165</td>
      <td>0.071048</td>
      <td>0.108094</td>
      <td>0.062928</td>
      <td>0.046435</td>
      <td>0.025121</td>
      <td>0.003552</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>Albany</td>
      <td>NY</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.111215</td>
      <td>0.064565</td>
      <td>0.078066</td>
      <td>0.069276</td>
      <td>0.056352</td>
      <td>0.080003</td>
      <td>0.071902</td>
      <td>0.061101</td>
      <td>0.054006</td>
      <td>0.094082</td>
      <td>0.101140</td>
      <td>0.101661</td>
      <td>0.034098</td>
      <td>0.007486</td>
      <td>0.009591</td>
      <td>0.005456</td>
      <td>Albuquerque</td>
      <td>NM</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.150833</td>
      <td>0.063611</td>
      <td>0.084792</td>
      <td>0.084514</td>
      <td>0.105903</td>
      <td>0.057778</td>
      <td>0.043542</td>
      <td>0.024583</td>
      <td>0.080417</td>
      <td>0.095417</td>
      <td>0.075556</td>
      <td>0.068958</td>
      <td>0.042639</td>
      <td>0.000000</td>
      <td>0.021458</td>
      <td>0.000000</td>
      <td>Allentown</td>
      <td>PA-NJ</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.084392</td>
      <td>0.086299</td>
      <td>0.077275</td>
      <td>0.135104</td>
      <td>0.043213</td>
      <td>0.134596</td>
      <td>0.061642</td>
      <td>0.051856</td>
      <td>0.037239</td>
      <td>0.120742</td>
      <td>0.054270</td>
      <td>0.070285</td>
      <td>0.011185</td>
      <td>0.022496</td>
      <td>0.000000</td>
      <td>0.009405</td>
      <td>Amarillo</td>
      <td>TX</td>
      <td>2006</td>
    </tr>
  </tbody>
</table>
</div>





```python
###############
# GINI INDEX
###############
gini_all = pd.DataFrame()
for year in range(2006, 2017):
    f = get_file_name(year, 'B19083')
    gini = pd.read_csv("data/gini/%s.csv" %f, encoding='Latin-1')
    # Don't need micro areas
    gini = slim_df(gini, ["HD01_VD01"])
    gini['year'] = year
    gini_all = append_df(gini, gini_all) 

# Clean Final Dataframes
gini_all = gini_all.rename(index=str,
                           columns={"HD01_VD01":"gini"})
gini_all['gini'] = gini_all['gini'].astype(float)
gini_all = match_crime(gini_all)
del gini_all['MSA']
gini_all.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>gini</th>
      <th>city_key</th>
      <th>state_key</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.443</td>
      <td>Abilene</td>
      <td>TX</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.533</td>
      <td>Aguadilla</td>
      <td>PR</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.445</td>
      <td>Akron</td>
      <td>OH</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.481</td>
      <td>Albany</td>
      <td>GA</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.405</td>
      <td>Albany</td>
      <td>NY</td>
      <td>2006</td>
    </tr>
  </tbody>
</table>
</div>





```python
#################
# Poverty Data
#################
pov_all = pd.DataFrame()
for year in range(2006, 2017):
    f = get_file_name(year, 'S1701')
    pov = pd.read_csv("data/poverty/%s.csv" %f, encoding='Latin-1')
    pov = slim_df(pov, ['HC03_EST_VC03', 'HC03_EST_VC05',
                       'HC03_EST_VC08', 'HC03_EST_VC09'])
    pov['year'] = year
    pov_all = append_df(pov, pov_all)
# Clean Final DataFrame
pov_all = pov_all.rename(index=str,
                        columns={'HC03_EST_VC03': 'under_18_pov',
                                'HC03_EST_VC05':'18_64_pov',
                                'HC03_EST_VC08':'male_pov',
                                'HC03_EST_VC09':'female_pov'})
pov_all = match_crime(pov_all)
del pov_all['MSA']
pov_all.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>under_18_pov</th>
      <th>18_64_pov</th>
      <th>male_pov</th>
      <th>female_pov</th>
      <th>city_key</th>
      <th>state_key</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>20.4</td>
      <td>14.9</td>
      <td>15.5</td>
      <td>16.1</td>
      <td>Abilene</td>
      <td>TX</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1</th>
      <td>67.4</td>
      <td>53.4</td>
      <td>56.3</td>
      <td>57.6</td>
      <td>Aguadilla</td>
      <td>PR</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>15.7</td>
      <td>12.5</td>
      <td>10.7</td>
      <td>14.6</td>
      <td>Akron</td>
      <td>OH</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>3</th>
      <td>31.0</td>
      <td>20.7</td>
      <td>20.7</td>
      <td>24.7</td>
      <td>Albany</td>
      <td>GA</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>13.2</td>
      <td>8.6</td>
      <td>8.7</td>
      <td>10.9</td>
      <td>Albany</td>
      <td>NY</td>
      <td>2006</td>
    </tr>
  </tbody>
</table>
</div>





```python
#################################
# Head of Household Information
#################################
house_all = pd.DataFrame()
for year in range(2006, 2017):
    f = get_file_name(year, 'B09005')
    house = pd.read_csv("data/house_head/%s.csv" %f, encoding='Latin-1')
    house = slim_df(house, [v for v in house.columns if "HD01" in v])
    house = house.loc[:, ["MSA", "city_key", "state_key",
                         "HD01_VD01", "HD01_VD03", "HD01_VD05",
                         "HD01_VD06"]]
    house['year'] = year
    house_all = append_df(house, house_all) 

# Clean Entire DataFrame
house_all = house_all.rename(index=str,
                             columns={'HD01_VD01': 'total',
                                     'HD01_VD03': 'married_house',
                                     'HD01_VD05': 'female_house',
                                     'HD01_VD06': 'male_house'})

house_all = convert_to_int(house_all,
                       ['total', 'married_house', 'female_house', 'male_house'])
house_all = create_proportions(house_all, ['married_house', 'female_house', 'male_house'], 'total')
house_all = match_crime(house_all)
del house_all['MSA']
house_all.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city_key</th>
      <th>state_key</th>
      <th>married_house</th>
      <th>female_house</th>
      <th>male_house</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abilene</td>
      <td>TX</td>
      <td>0.661882</td>
      <td>0.253355</td>
      <td>0.007038</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aguadilla</td>
      <td>PR</td>
      <td>0.625417</td>
      <td>0.324838</td>
      <td>0.004308</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Akron</td>
      <td>OH</td>
      <td>0.686886</td>
      <td>0.246946</td>
      <td>0.008047</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albany</td>
      <td>GA</td>
      <td>0.505709</td>
      <td>0.429540</td>
      <td>0.000000</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Albany</td>
      <td>NY</td>
      <td>0.668823</td>
      <td>0.251285</td>
      <td>0.013663</td>
      <td>2006</td>
    </tr>
  </tbody>
</table>
</div>





```python
#################
# Education Data
#################
edu_all = pd.DataFrame()
for year in range(2006,2017):
    f = get_file_name(year, 'S1501')
    edu = pd.read_csv("data/education/%s.csv" %f, encoding='Latin-1')
    if 2015 <= year <= 2016:
        edu = slim_df(edu, ['HC02_EST_VC03', 'HC02_EST_VC04', 'HC02_EST_VC09','HC02_EST_VC10', 'HC02_EST_VC11'])
    elif 2010 <= year <= 2014:
        edu = slim_df(edu,['HC01_EST_VC02', 'HC01_EST_VC03', 'HC01_EST_VC08','HC01_EST_VC09', 'HC01_EST_VC10'])
    else:
        edu = slim_df(edu, ['HC01_EST_VC02', 'HC01_EST_VC03', 'HC01_EST_VC07','HC01_EST_VC08', 'HC01_EST_VC09'])
    
    edu['year'] = year
    edu.columns=['no_hs_18_24','hs_18_24','no_9th_25_ovr','no_hs_25_ovr','hs_25_ovr','MSA','city_key','state_key','year']
    edu_all = append_df(edu, edu_all)

#Trim Final Dataframe
edu_all = match_crime(edu_all)
del edu_all['MSA']
edu_all.head()

```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>no_hs_18_24</th>
      <th>hs_18_24</th>
      <th>no_9th_25_ovr</th>
      <th>no_hs_25_ovr</th>
      <th>hs_25_ovr</th>
      <th>city_key</th>
      <th>state_key</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10.8</td>
      <td>30.2</td>
      <td>7.5</td>
      <td>14.5</td>
      <td>28.6</td>
      <td>Abilene</td>
      <td>TX</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1</th>
      <td>23.4</td>
      <td>30.6</td>
      <td>32.7</td>
      <td>11.7</td>
      <td>25.0</td>
      <td>Aguadilla</td>
      <td>PR</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>11.6</td>
      <td>34.1</td>
      <td>2.4</td>
      <td>8.9</td>
      <td>34.4</td>
      <td>Akron</td>
      <td>OH</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>3</th>
      <td>29.1</td>
      <td>22.1</td>
      <td>7.0</td>
      <td>12.2</td>
      <td>31.5</td>
      <td>Albany</td>
      <td>GA</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>11.9</td>
      <td>29.7</td>
      <td>3.2</td>
      <td>6.9</td>
      <td>30.5</td>
      <td>Albany</td>
      <td>NY</td>
      <td>2006</td>
    </tr>
  </tbody>
</table>
</div>





```python
############
# Race Data
############
race_all = pd.DataFrame()
for year in range(2006, 2017):
    f = get_file_name(year, 'B02001')
    race = pd.read_csv("data/race/%s.csv" %f, encoding='Latin-1')
    race = slim_df(race, [v for v in race.columns if "HD01" in v])
    race = race.loc[:, ['MSA', 'city_key', 'state_key',
                       'HD01_VD01','HD01_VD02',
                       'HD01_VD03', 'HD01_VD05']]
    
    race['year'] = year
    race_all = append_df(race, race_all)

# Proccess Final Data Frame
race_all =  race_all.rename(index=str,
                            columns={'HD01_VD01':'total',
                                    'HD01_VD02': 'white',
                                    'HD01_VD03': 'black',
                                    'HD01_VD05': 'asian'})

race_all = convert_to_int(race_all, ['total', 'white','black','asian'] )
race_all = create_proportions(race_all, ['white', 'black', 'asian'], 'total')
# Match Crime Data
race_all = match_crime(race_all)
del race_all['MSA']
race_all.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city_key</th>
      <th>state_key</th>
      <th>white</th>
      <th>black</th>
      <th>asian</th>
      <th>year</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abilene</td>
      <td>TX</td>
      <td>0.741838</td>
      <td>0.068295</td>
      <td>0.014292</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aguadilla</td>
      <td>PR</td>
      <td>0.896527</td>
      <td>0.019961</td>
      <td>0.000758</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Akron</td>
      <td>OH</td>
      <td>0.844866</td>
      <td>0.116880</td>
      <td>0.017715</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albany</td>
      <td>GA</td>
      <td>0.485957</td>
      <td>0.494136</td>
      <td>0.006355</td>
      <td>2006</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Albany</td>
      <td>NY</td>
      <td>0.867237</td>
      <td>0.070216</td>
      <td>0.030761</td>
      <td>2006</td>
    </tr>
  </tbody>
</table>
</div>





```python
# Bring Everything Together
census_df = race_all.copy()

merge_df = lambda df: census_df.merge(df,
                                     how='outer',
                                     on=['city_key','state_key','year'],
                                     indicator=True)

str_list = ['Employment', 'Age', 'Head of House','Education', 'Gini', 'Poverty']
df_list = [emp_all, age_all, house_all, edu_all, gini_all, pov_all]

for i, df in enumerate(df_list):
    census_df = merge_df(df)
    #print("%s Merge Stats" %str_list[i])
    #print(census_df['_merge'].value_counts())
    del census_df['_merge']
    
'''

Household had ones where it looks like there may be mismatches but code below checked it

Code to check merges
names = census_df.loc[census_df._merge != "both", ['city_key', 'state_key', '_merge']]
names = names.sort_values['city_key', 'state_key']
names = names.drop_duplicates()
print(names.shape[0])
names.head(50)
'''
```





    '\n\nHousehold had ones where it looks like there may be mismatches but code below checked it\n\nCode to check merges\nnames = census_df.loc[census_df._merge != "both", [\'city_key\', \'state_key\', \'_merge\']]\nnames = names.sort_values[\'city_key\', \'state_key\']\nnames = names.drop_duplicates()\nprint(names.shape[0])\nnames.head(50)\n'





```python
#####################
# Bring in BEA Data
#####################
bea_gdp = pd.read_csv("data/BEA_real_GDP_pc.csv",skiprows=[0,1,2], header=1)
del bea_gdp['Fips']
bea_gdp= bea_gdp.iloc[1:, :].rename(index=str, columns={"Area": 'MSA'})
bea_gdp = pd.melt(bea_gdp, id_vars=["MSA"], var_name='year', value_name='real_pc_gdp')
bea_gdp = bea_gdp.loc[bea_gdp.MSA.notnull(), :]
bea_gdp['year'] = bea_gdp['year'].astype(int)
bea_gdp = bea_gdp.loc[bea_gdp.year >= 2006, :]

# Get rid of MSA in paranthesis
bea_gdp['MSA'] = bea_gdp['MSA'].str.replace(r"\(.*\)","")
bea_gdp = split_MSA(bea_gdp)
# Need to manually fix this one so it will merge
bea_gdp.loc[bea_gdp.city_key.str.contains("Louisville"), 'city_key'] = 'Louisville'

del bea_gdp['MSA']
```




```python
census_df = merge_df(bea_gdp)
```




```python
# Check to make sure that there were no typos
names = census_df.loc[census_df._merge != "both", ['city_key', 'state_key', '_merge']]
names = names.drop_duplicates()
print(names.shape[0])
del census_df["_merge"]
```


    49




```python
census_df.head()
```





<div>
<style>
    .dataframe thead tr:only-child th {
        text-align: right;
    }

    .dataframe thead th {
        text-align: left;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>city_key</th>
      <th>state_key</th>
      <th>white</th>
      <th>black</th>
      <th>asian</th>
      <th>year</th>
      <th>unemp_16_ovr</th>
      <th>unemp_16_19</th>
      <th>unemp_female</th>
      <th>median_age</th>
      <th>...</th>
      <th>hs_18_24</th>
      <th>no_9th_25_ovr</th>
      <th>no_hs_25_ovr</th>
      <th>hs_25_ovr</th>
      <th>gini</th>
      <th>under_18_pov</th>
      <th>18_64_pov</th>
      <th>male_pov</th>
      <th>female_pov</th>
      <th>real_pc_gdp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Abilene</td>
      <td>TX</td>
      <td>0.741838</td>
      <td>0.068295</td>
      <td>0.014292</td>
      <td>2006</td>
      <td>6.6</td>
      <td>23.1</td>
      <td>5.2</td>
      <td>34.4</td>
      <td>...</td>
      <td>30.2</td>
      <td>7.5</td>
      <td>14.5</td>
      <td>28.6</td>
      <td>0.443</td>
      <td>20.4</td>
      <td>14.9</td>
      <td>15.5</td>
      <td>16.1</td>
      <td>33978.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Aguadilla</td>
      <td>PR</td>
      <td>0.896527</td>
      <td>0.019961</td>
      <td>0.000758</td>
      <td>2006</td>
      <td>20.6</td>
      <td>56.8</td>
      <td>19.1</td>
      <td>35.3</td>
      <td>...</td>
      <td>30.6</td>
      <td>32.7</td>
      <td>11.7</td>
      <td>25.0</td>
      <td>0.533</td>
      <td>67.4</td>
      <td>53.4</td>
      <td>56.3</td>
      <td>57.6</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Akron</td>
      <td>OH</td>
      <td>0.844866</td>
      <td>0.116880</td>
      <td>0.017715</td>
      <td>2006</td>
      <td>6.3</td>
      <td>23.0</td>
      <td>4.8</td>
      <td>37.9</td>
      <td>...</td>
      <td>34.1</td>
      <td>2.4</td>
      <td>8.9</td>
      <td>34.4</td>
      <td>0.445</td>
      <td>15.7</td>
      <td>12.5</td>
      <td>10.7</td>
      <td>14.6</td>
      <td>42081.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Albany</td>
      <td>GA</td>
      <td>0.485957</td>
      <td>0.494136</td>
      <td>0.006355</td>
      <td>2006</td>
      <td>10.0</td>
      <td>31.0</td>
      <td>10.1</td>
      <td>34.3</td>
      <td>...</td>
      <td>22.1</td>
      <td>7.0</td>
      <td>12.2</td>
      <td>31.5</td>
      <td>0.481</td>
      <td>31.0</td>
      <td>20.7</td>
      <td>20.7</td>
      <td>24.7</td>
      <td>32657.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Albany</td>
      <td>NY</td>
      <td>0.867237</td>
      <td>0.070216</td>
      <td>0.030761</td>
      <td>2006</td>
      <td>5.4</td>
      <td>16.8</td>
      <td>4.4</td>
      <td>38.2</td>
      <td>...</td>
      <td>29.7</td>
      <td>3.2</td>
      <td>6.9</td>
      <td>30.5</td>
      <td>0.405</td>
      <td>13.2</td>
      <td>8.6</td>
      <td>8.7</td>
      <td>10.9</td>
      <td>49549.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 30 columns</p>
</div>



## Load Crime Data



```python
# THIS CODE ONLY NEEDS TO BE RUN ONCE TO BRING IN ALL OF THE EXCEL FILES
version='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.75 Safari/537.36'
test=urllib.request.URLopener()
test.addheader('User-Agent',version)
for year in range(2006, 2017):
    print("Pulling: %i" %year)
    test.retrieve(url=fbi_url_generator(year),filename='crime_%i.xls' %year)

```


    Pulling: 2006
    Pulling: 2007
    Pulling: 2008
    Pulling: 2009
    Pulling: 2010
    Pulling: 2011




```python
df_allyears = pd.DataFrame()
for year in range(2006, 2017):
    df = pd.read_excel("crime_%i.xls" %year,skiprows=[0,1],header=1)

    #######
    # NOTE - misc column has msa population, city population and estimate percentage
    ######
    df=df.iloc[:,0:12] 
    df.columns=['MSA', 'counties','misc', 'violent_crime','mur_mans', 'rape', 'robbery',
                'assault', 'property', 'burglary', 'larceny','mv_theft']
    
    df['counties'].replace(' ',np.nan, inplace=True)

    # Drop footnotes
    footnotes = df['MSA'].str[0].str.isdigit().fillna(False)
    df = df.loc[~footnotes, :]
    
    #Drop blank rows
    df = df.dropna(how='all')

    # Get rid of numbers in MSA
    df['MSA'] = df['MSA'].str.replace('\d+', '')
    # Set empty columns to NaN for MSA
    df['MSA'] = df['MSA'].replace(' ', np.nan, regex=False)
    
    # Sometimes city  names get put in MSA column
    # Messes up carry forward
    df.loc[df['MSA'].str.contains("City of").fillna(False), "MSA"] = np.nan

    # Carry MSA name forward to fill in for all cells
    df.loc[:,'MSA'] = df.loc[:, 'MSA'].fillna(method='ffill')

    ##############
    # POPULATION - grab population and fill in for all MSA
    ##############
    pop_row = df.counties.isnull()
    pop = df.loc[pop_row, ["MSA", 'misc']]
    pop = pop.rename(index=str, columns={'misc': 'msa_pop'})
    
    # Merge population back in
    df = df.loc[~pop_row, :]
    df = df.merge(pop, how='outer', on='MSA')

    ################
    # Descriptions - don't need county descriptions 
    ################
    df = df.loc[df.counties.str.contains("Includes") == False, :]


    ###########################################
    # GOING LONG TO WIDE FOR CRIME VARIABLES
    ###########################################
    crime_vars = ['violent_crime','mur_mans', 'rape', 'robbery',
                  'assault', 'property', 'burglary', 'larceny','mv_theft']

    #########
    # CITIES
    #########
    city_vars = ['MSA', 'counties', 'misc'] + crime_vars
    # Split data Frame
    cities = df.counties.str.contains("City")
    city_df = df.loc[cities, city_vars]
    city_df = city_df.rename(index=str, columns={'misc': 'city_pop'})
    
    # Grab largest city for each MSA and merge back on
    city_df = city_df.sort_values(['MSA','city_pop'], ascending=False)
    large_city = city_df.groupby('MSA').first().reset_index()

    # Rename crime variables to denote city only crime 
    large_city.columns = ['MSA', 'counties', 'city_pop'] + ['city_' + i for i in crime_vars]
    large_city = large_city.rename(index=str, columns={'counties':'largest_city'})
    # Get rid of "City of"
    large_city.loc[:,'largest_city'] = large_city.loc[:, 'largest_city'].str.replace('City of','')
    
    # Merge back to main dataframe
    df = df.loc[~cities, ]
    df = df.merge(large_city, how='outer', on='MSA')

    ###############
    # CRIME RATE
    ###############
    rates = df.counties.str.contains("Rate per")
    rate_vars = ['MSA'] + crime_vars
    rates_df = df.loc[rates, rate_vars]
    rates_df.columns = ['MSA'] + ['rate_' + i for i in crime_vars]

    df = df.loc[~rates, :]
    df = df.merge(rates_df, how='outer', on='MSA')

    ########################
    # MSA-WIDE CRIME STATS
    ########################

    # If the entire MSA reported then there is just one row of numbers
    # If the entire MSA did not report, then there are two rows
            # first row is areas that reported
            # second report is an estimated total
    # We are going to grab the estimates total so our data
    # reflects all areas for all MSA

    # Create Flag for those that do not have complete coverage
    # and are thus estimates
    mins = df.groupby('MSA').misc.min().reset_index()
    mins.columns = ['MSA', 'min_coverage']
    df = df.merge(mins, how='outer', on='MSA')
    df['estimate'] = 0
    df.loc[df.min_coverage < 1, 'estimate'] = 1
    del df['min_coverage']

    # Now only keeping rows with coverage = 1
    # will either be all area or the estimate for all area
    df = df.loc[df.misc == 1, :]

    # Now no longer need coverage or whether its estimate or not
    del df['misc']
    del df['counties']
    
    df['year'] = year
    
    # Append to existing Frame
    df_allyears = append_df(df, df_allyears)
```




```python
df_allyears = df_allyears.sort_values(["MSA", 'year'])
```




```python

#Generate the city
df_allyears['city_key'] = df_allyears['MSA'].str.replace(' M.S.A.','').str.split(",").str[0]
df_allyears['city_key'] = df_allyears['city_key'].str.split("-").str[0].str.strip()
df_allyears['state_key']=df_allyears['MSA'].str.replace(' M.S.A.','').str.split(",").str[1].str.strip().str.replace(' M.S.A','')
```




```python
##Cleanse Crime Data
df_allyears=df_allyears[~df_allyears['MSA'].str.contains(' M.D.')]
df_allyears.loc[df_allyears['state_key']=='Puerto Rico','state_key']='PR'
df_allyears.loc[df_allyears['MSA'].str.contains('Texarkana'),'state_key']='AR-TX'
df_allyears.loc[df_allyears['city_key']=='Worcester','state_key']='MA-CT'
df_allyears.loc[df_allyears['city_key']=='Steubenville','city_key']='Weirton'
df_allyears.loc[df_allyears['city_key']=='Steubenville','state_key']='WV-OH'
df_allyears.loc[df_allyears['city_key']=='Honolulu','city_key']='Urban Honolulu'
df_allyears.loc[df_allyears['MSA'].str.contains('Scranton'),'city_key']='Scranton'
df_allyears.loc[df_allyears['MSA'].str.contains('Sarasota'),'city_key']='North Port'
df_allyears.loc[df_allyears['MSA'].str.contains('Santa Maria'),'city_key']='Santa Maria'
df_allyears.loc[df_allyears['MSA'].str.contains('Salisbury'),'state_key']='MD-DE'
df_allyears.loc[df_allyears['MSA'].str.contains('Sacramento'),'city_key']='Sacramento'
df_allyears.loc[df_allyears['MSA'].str.contains('Myrtle Beach'),'state_key']='SC-NC'
df_allyears.loc[df_allyears['MSA'].str.contains('Louisville'),'city_key']='Louisville'
df_allyears.loc[df_allyears['MSA'].str.contains('Homosassa'),'city_key']='Homosassa Springs'
df_allyears.loc[df_allyears['MSA'].str.contains('Crestview'),'city_key']='Crestview'
```


## Merge Crime & Census



```python
#Merge Crime and census data
final_df = df_allyears.merge(census_df, how='left', on=['city_key','state_key','year'])
```




```python
# Function to convert to float otherwise set to NaN
def f(x):
    try:
        return np.float(x)
    except:
        return np.nan
```




```python
# Look at final Cleaning of data-types
float_cols = final_df.columns.difference(["MSA", "city_key", "state_key","year", "largest_city"])
for v in float_cols:
    try:
        final_df[v] = final_df[v].astype(float)
    except:
        continue
# Assault Rate Needs to be fixed
# One Value is missing
final_df.loc[final_df.rate_assault == " ", 'rate_assault'] = np.nan
final_df["rate_assault"] = final_df["rate_assault"].astype(float)
```




```python
#Create a join key
final_df['join_key'] = final_df['city_key'].str.cat(final_df['state_key'],sep='-')
#Add columns for OHE
final_df['year_ohe'] = final_df['year']
final_df['state_ohe'] = final_df['state_key']
final_df['join_ohe'] = final_df['join_key']
#One hot encode join key, state key and year
final_df = pd.get_dummies(final_df,prefix='year',columns=['year_ohe']) #Not dropping one column since year has missing values
final_df = pd.get_dummies(final_df,prefix=['MSA','state'],columns=['join_ohe','state_ohe'],drop_first=True)
```




```python
final_df.to_json('output/final.json')
```

