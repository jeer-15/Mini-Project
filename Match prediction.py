#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[3]:


Indian_perf = pd.read_csv("Indian performnces.csv")
Indian_team= pd.read_csv("Indian Team.csv")


# In[4]:


Indian_perf.head(10)


# In[5]:


Indian_team.head()


# In[6]:


Indian_perf.describe()


# In[7]:


Indian_team.describe()


# In[8]:


Indian_perf.isnull().sum()


# In[9]:


Indian_team.isnull().sum()


# In[10]:


Indian_perf=Indian_perf.fillna('')


# In[11]:


Indian_perf.isnull().sum()


# In[12]:


Indian_team=Indian_team.fillna('')


# In[13]:


Indian_team.isnull().sum()


# In[14]:


#EDA
#Stacked bar chart
#Create a contingency table
contingency_table = pd.crosstab(Indian_perf['Venue'], Indian_perf['Result'])


# In[2]:


#Visualize
# Set custom width and height for the figure
fig, ax = plt.subplots(figsize=(15, 20))
contingency_table.plot(kind='boxplot', stacked=True, ax=ax)
plt.title("Ground vs. Result")
plt.ylabel("Venue")
plt.xlabel("Count")
plt.legend(title="Result")

plt.show()


# In[16]:


#Get unique values
unique_grounds = Indian_perf['Venue'].unique()
for Ground in unique_grounds:
    print(Ground)


# In[17]:


#Define a list of known indian grounds
Indian_grounds = ['Mirpur', 'Bengaluru', 'Delhi', 'Nagpur', 'Chennai', 'Ahmedabad', 'Mohali', 'Wankhede', 'Hyderabad', 
                  'Eden Gardens', 'Cuttack', 'Visakhapatnam', 'Indore', 'Rajkot', 'Kochi', 'Ranchi', 'Dharamsala',
                 'Pune', 'Jaipur', 'Kanpur', 'Guwahati', 'Thiruvananthapuram', 'Lucknow', 'Brabourne', 'Raipur']


# In[18]:


india_ground = Indian_perf[Indian_perf['Venue'].isin(Indian_grounds)]
away_ground = Indian_perf[~Indian_perf['Venue'].isin(Indian_grounds)]


# In[19]:


#Calculate the winnings
india_home_wins = india_ground[india_ground['Result'] == 'won']
away_home_wins = away_ground[away_ground['Result'] == 'won']


# In[20]:


# Create a bar chart to compare the results
plt.figure(figsize=(6,4))
plt.bar(['India Ground', 'Away Ground'], [len(india_home_wins), len(away_home_wins)], color=['grey', 'red'])
plt.title('Home Team Wins vs Away Team Wins')
plt.xlabel('Ground Type')
plt.ylabel('Number of Wins')
plt.show()


# In[21]:


#Comparing the two attributes
#Opposition vs Result
# Group the data by the "opposition" and "result" columns and count the occurrences
result_counts = Indian_perf.groupby(['Opponent', 'Result']).size().unstack(fill_value=0)

# Create a bar chart to visualize the results
result_counts.plot(kind='bar', stacked=True, figsize=(5, 4))
plt.xlabel('Opposition Teams')
plt.ylabel('Number of Matches')
plt.title('Comparison of Results by Opposition')
plt.legend(title='Result', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.show()


# In[1]:


# Create a countplot to compare results based on Opponent and Ground
plt.figure(figsize=(12, 8))
sns.set(style="jeer")


# In[23]:


#Visualization
sns.countplot(data=Indian_perf, x='Opponent', hue='Result', palette='Set1')
# Rotate x-axis labels for better readability
plt.xticks(rotation=90) 
plt.xlabel('Opponent and Ground')
plt.ylabel('Number of Matches')
plt.title('Comparison of Results by Opponent and Ground')

plt.show()


# In[24]:


#Comparison of Innings and Results
innings_result_counts = Indian_perf.groupby(['Inns', 'Result']).size().unstack(fill_value=0)


# In[25]:


# Create a pie chart for the 1st Innings
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.pie(innings_result_counts.loc[1], labels=innings_result_counts.columns, autopct='%1.1f%%', startangle=90)
plt.title('1st Innings Result')


# In[26]:


# Create a pie chart for the 2nd Innings
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 2)
plt.pie(innings_result_counts.loc[2], labels=innings_result_counts.columns, autopct='%1.1f%%', startangle=90)
plt.title('2nd Innings Result')


# In[27]:


# Create a grouped bar chart
fig, ax = plt.subplots(figsize=(10, 6))

# Unique innings and results
unique_innings = Indian_perf['Inns'].unique()
unique_results = Indian_perf['Result'].unique()

bar_width = 0.2
index = range(len(unique_results))

#Defining the color
colors = ['red', 'green', 'blue']

# Create a bar chart for innings and results
for i, inning in enumerate(unique_innings):
    ground_counts = [len(Indian_perf[(Indian_perf['Inns'] == inning) & (Indian_perf['Result'] == r)]) for r in unique_results]
    plt.bar([pos + i * bar_width for pos in index], ground_counts, bar_width, label=f'Inns {inning}', color=colors[i],)

plt.xlabel('Result')
plt.ylabel('Count')
plt.xticks([pos + bar_width for pos in index], unique_results)
plt.legend(title='Innings')
plt.title('Comparison of Innings and Results')
plt.show()


# In[28]:


#Calculating the maximum score
player_stats = Indian_team.groupby('Name')['HS_modified'].max().reset_index()

# Sort the players by their highest scores (optional)
player_stats = player_stats.sort_values(by='HS_modified', ascending=False)

# Display or analyze the results
print(player_stats)


# In[29]:


#Batting performance
#Barchart
plt.figure(figsize=(16,8))
plt.bar(Indian_team['Name'], Indian_team['Runs'])
plt.xlabel('Name')
plt.ylabel('Runs')
plt.show()


# In[30]:


#Barchart
plt.figure(figsize=(16,8))
plt.bar(Indian_team['Name'], Indian_team['HS_modified'])
plt.xlabel('Name')
plt.ylabel('HS_modified')
plt.show()


# In[31]:


#Barchart
plt.figure(figsize=(16,8))
plt.bar(Indian_team['Name'], Indian_team['NO'])
plt.xlabel('Name')
plt.ylabel('NO')
plt.show()


# In[32]:


#Barchart
plt.figure(figsize=(16,8))
plt.bar(Indian_team['Name'], Indian_team['Ave'])
plt.xlabel('Name')
plt.ylabel('Ave')
plt.show()


# In[33]:


#Barchart
plt.figure(figsize=(16,8))
plt.bar(Indian_team['Name'], Indian_team['SR'])
plt.xlabel('Name')
plt.ylabel('SR')
plt.show()


# In[34]:


#Bowling performance
#Barchart
plt.figure(figsize=(16,8))
plt.bar(Indian_team['Name'], Indian_team['wkts'])
plt.xlabel('Name')
plt.ylabel('wkts')
plt.show()


# In[35]:


#Barchart
plt.figure(figsize=(16,8))
plt.bar(Indian_team['Name'], Indian_team['Bowl Ave'])
plt.xlabel('Name')
plt.ylabel('Bowl Ave')
plt.show()


# In[36]:


#Barchart
plt.figure(figsize=(16,8))
plt.bar(Indian_team['Name'], Indian_team['Eco'])
plt.xlabel('Name')
plt.ylabel('Eco')
plt.show()


# In[37]:


from sklearn.preprocessing import LabelEncoder


# In[38]:


le = LabelEncoder()
Indian_perf['venue_encoded'] = le.fit_transform(Indian_perf['Venue'])
Indian_perf['opposition_encoded'] = le.fit_transform(Indian_perf['Opponent'])


# In[39]:


# Select features and target
X = Indian_perf[['venue_encoded', 'opposition_encoded', 'Inns']]
y = Indian_perf['Result']


# In[40]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# In[41]:


# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[42]:


# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[43]:


# Create and train the logistic regression model
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)


# In[44]:


# Make predictions on the test set
y_pred = logistic_regression_model.predict(X_test)


# In[45]:


# Evaluate the model
accuracy = logistic_regression_model.score(X_test, y_test)
print(f"Accuracy: {accuracy:.2f}")


# In[46]:


#Get unique values
unique_opponent = Indian_perf['Opponent'].unique()
for Opponent in unique_opponent:
    print(Opponent)


# In[47]:


#Replace function
# Create a mapping of opposition teams to numbers
opposition_mapping = {
    'v South Africa': 1,
    'v Bangladesh': 2,
    'v England': 3,
    'v Ireland': 4,
    'v Netherlands': 5,
    'v West Indies': 6, 
    'v Australia': 7,
    'v Pakistan': 8,
    'v Sri Lanka': 9,
    'v Zimbabwe': 10,
    'v New Zealand': 11,
    'v Afghanistan': 12,
    'v U.A.E.': 13,
    'v Hong Kong': 14,
    'v Nepal': 15
}


# In[48]:


# Replace values in the 'Opposition' column
Indian_perf['Opponent'] = Indian_perf['Opponent'].replace(opposition_mapping)


# In[49]:


print(Indian_perf)


# In[50]:


venue_mapping = {
    'Durban': 1,
'Johannesburg': 2,
'Cape Town': 3,
'Gqeberha': 4,
'Centurion': 5,
'Mirpur': 6,
'Bengaluru': 7,
'Delhi': 8,
'Nagpur': 9,
'Chennai': 10,
'Ahmedabad': 11,
'Mohali': 12,
'Wankhede': 13,
'Port of Spain': 14,
'North Sound': 15,
'Kingston': 16,
'Chester-le-Street': 17,
'Southampton': 18,
'The Oval': 19,
'Lords': 20,
'Cardiff': 21,
'Hyderabad': 22,
'Eden Gardens': 23,
'Cuttack': 24,
'Visakhapatnam': 25,
'Indore': 26,
'Melbourne': 27,
'Perth': 28,
'Adelaide': 29,
'Brisbane': 30,
'Sydney': 31,
'Hobart': 32,
'Hambantota': 33,
'Colombo (RPS)': 34,
'Pallekele': 35,
'Rajkot': 36,
'Kochi': 37,
'Ranchi': 38,
'Dharamsala': 39,
'Birmingham': 40, 
'Harare': 41,
'Bulawayo': 42,
'Pune': 43,
'Jaipur': 44,
'Kanpur': 45,
'Napier': 46,
'Hamilton': 47,
'Auckland': 48,
'Wellington': 49,
'Fatullah': 50,
'Nottingham': 51,
'Leeds': 52,
'Canberra': 53,
'Dambulla': 54,
'Dubai (DSC)': 55,
'Guwahati': 56,
'Brabourne': 57,
'Thiruvananthapuram': 58,
'Mount Maunganui': 59,
'Manchester': 60,
'Providence': 61,
'Paarl': 62,
'Lucknow': 63,
'Christchurch': 64,
'Chattogram': 65,
'Raipur': 66,
'Bridgetown': 67,
'Tarouba': 68
}


# In[51]:


# Replace values in the 'Opposition' column
Indian_perf['Venue'] = Indian_perf['Venue'].replace(venue_mapping)


# In[52]:


print(Indian_perf)


# In[69]:


# Take user input for prediction
user_opponent = input("Enter opponent(1 to 15): ")
user_inns = int(input("Enter innings (1 or 2): "))
user_venue = input("Enter venue(1 to 68): ")


# In[62]:


user_input = [[user_opponent, user_inns, user_venue]]


# In[63]:


# Check data types of user input
print(f"User Input Data Types: {type(user_opponent)}, {type(user_inns)}, {type(user_venue)}")


# In[64]:


# Standardize the user input using the scaler
user_input_standardized = scaler.transform(user_input)


# In[65]:


# Check the standardized user input
print(f"Standardized User Input: {user_input_standardized}")


# In[66]:


# Make probability predictions for the user input
probability_prediction = logistic_regression_model.predict_proba(user_input_standardized)


# In[67]:


# Extract the probability of India winning (class 1)
percentage_chance = probability_prediction[0][1] * 100


# In[68]:


print(f"The predicted percentage chance of India winning is: {percentage_chance:.2f}%")


# In[ ]:




