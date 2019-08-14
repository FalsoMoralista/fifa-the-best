import pandas as pd
from sklearn.preprocessing import LabelEncoder, LabelBinarizer
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split 


def getDataFrame(values, target):
    print("\nextracting feature names")
    feature_names = list(values.columns.values)
    print("\ncreating selector objects")
    skbest = SelectKBest(mutual_info_regression, k = 20)
    skbest_fr = SelectKBest(f_regression, k = 20)
    print("\nselecting attributes")
    new_values = skbest.fit(values, target)
    nv_fr = skbest_fr.fit(values, target)
    print("\ngetting selected attributes")
    features_selected = new_values.get_support()
    features_selected_fr = nv_fr.get_support()
    
    new_values = skbest.transform(values)
    
    print("\ncreating feature selected set")
    new_features = set() # The list of your K best features
    new_featues_fr = set()
    
    for bool, feature in zip(features_selected, feature_names):
        if bool:
            new_features.add(feature)
    for bool, feature in zip(features_selected_fr, feature_names):
        if bool:
            new_featues_fr.add(feature)
    print("\ngetting intersected attributes")
    inter = new_features.intersection(new_featues_fr)
    dataframe = pd.DataFrame(new_values, columns=new_features)
    dataframe = dataframe[inter]
    return dataframe
    

#loading original dataset
data = pd.read_csv('completo.csv')
#count number of nans per attribute
nan_columns = data.isnull().sum(axis=0)

#handling empty values
club_mode = data['club'].mode()
league_mode = data['league'].mode()
data['eur_release_clause'] = data['eur_release_clause'].fillna(0)
data['league'] = data['league'].fillna(league_mode)
data['club'] = data['club'].fillna(club_mode)


#remove some attributes by inspection
inspected_attributes = ['club_logo', 'flag', 'photo', 'ID']
data = data.drop(columns=inspected_attributes)

#encondering categorical attributes
for column in data.columns:
    if data[column].dtype == type(object):
        data[column] = data[column].astype(str)
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])    
    elif data[column].dtype == bool:
        lb = LabelBinarizer()
        data[column] = lb.fit_transform(data[column])
        
#dropping target attributes
target_columns = ['eur_value','overall', 'potential']
data_csv = data.drop(columns=target_columns)
data_csv.to_csv('novo.csv')

#get all goalkeepers
goalkeepers = data.loc[data['gk'].notnull()]
#other players attributes
goalkeepers_removable_attributes = ['rs', 'rw', 'rf', 'ram', 'rcm', 
                                    'rm', 'rdm', 'rcb', 'rb', 'rwb',
                                    'st', 'lw', 'cf', 'cam', 'cm', 
                                    'lm', 'cdm', 'cb', 'lb', 'lwb',
                                    'ls', 'lf', 'lam', 'lcm', 'ldm',
                                    'lcb']
#remove other players attributes
goalkeepers = goalkeepers.drop(columns=goalkeepers_removable_attributes)
goalkeepers_target = goalkeepers[target_columns]
goalkeepers = goalkeepers.drop(columns=target_columns)

#other players
other_players = data.loc[data['gk'].isnull()]
#remove goalkeeper attribute
other_players = other_players.drop(columns=['gk'])
players_target = other_players[target_columns]
other_players = other_players.drop(columns=target_columns)

#feature selection 
new_other_players_eur_value = getDataFrame(other_players, players_target['eur_value'])
new_other_players_overall = getDataFrame(other_players, players_target['overall'])
new_other_players_potential = getDataFrame(other_players, players_target['potential'])

new_goalkeeperss_eur_value = getDataFrame(goalkeepers, goalkeepers_target['eur_value'])
new_goalkeeperss_overall = getDataFrame(goalkeepers, goalkeepers_target['overall'])
new_goalkeeperss_potential = getDataFrame(goalkeepers, goalkeepers_target['potential'])

#regression classifiers

gk_eur_value_dtr = DecisionTreeRegressor(criterion='mse', min_samples_split=300,min_samples_leaf=100, max_depth=6)
gk_overall_dtr = DecisionTreeRegressor(criterion='mae', min_samples_split=300, min_samples_leaf=100, max_depth=6)
gk_potential_dtr = DecisionTreeRegressor(criterion='mae', min_samples_split=300,min_samples_leaf=100, max_depth=6)

op_eur_value_dtr = DecisionTreeRegressor(criterion='mse', min_samples_split=3500,min_samples_leaf=1500, max_depth=6)
op_overall_dtr = DecisionTreeRegressor(criterion='mae', min_samples_split=3500,min_samples_leaf=1500, max_depth=6)
op_potential_dtr = DecisionTreeRegressor(criterion='mae', min_samples_split=3500,min_samples_leaf=1500, max_depth=6)

linear_regression = LinearRegression(fit_intercept = True, copy_X=True)
gk_knnRegressor = KNeighborsRegressor(n_neighbors=20, weights='distance', algorithm='kd_tree')
op_knnRegressor = KNeighborsRegressor(n_neighbors=150, weights='distance', algorithm='kd_tree')

#spliting data

X_train, X_test, y_train, y_test = train_test_split(  
    data, target, test_size = 0.3, random_state = 100) 






