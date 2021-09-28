import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

#Импорт данных обучающей выборки
df_train=pd.read_csv('gold_recovery_train_new.csv')
df_train_clear=df_train.fillna(value=None, method="ffill")
print(df_train_clear.isna().sum()*100/len(df_train))
#Импорт данных тестовой выборки
df_test=pd.read_csv('gold_recovery_test.csv')
df_test=df_test.fillna(value=None, method="ffill")
print(df_test.isna().sum()*100/len(df_test))
#Импорт данных полной выборки выборки
df_full=pd.read_csv('/datasets/gold_recovery_full_new.csv')
df_full=df_full.fillna(value=None, method="ffill")
print(df_full.isnull().sum()*100/len(df_full))

#анализ параметров, не используемых в тестовой выборке
df_traincs=df_train.columns
df_testcs=df_test.columns
df_diff=[]
for train in df_traincs:
    if train not in df_testcs:
        df_diff.append(train)
print(df_diff)

#Очистка данных
df_train_clear1=df_train
df_train_clear1=df_train.dropna()

#Оценка эффективности обогащения
F=df_train_clear1['rougher.input.feed_au']
print(F.isna().value_counts())
C=df_train_clear1['rougher.output.concentrate_au']
print(C.isna().value_counts())
T=df_train_clear1['rougher.output.tail_au']
print(T.isna().value_counts())
A=C*(F-T)
print(A.isna().value_counts())
B=F*(C-T)
print(B.isna().value_counts())
df_train_clear1['rougher_recovery']=(A/B)*100
print(df_train_clear1['rougher_recovery'].isna().value_counts())
print(mean_absolute_error(df_train_clear1['rougher.output.recovery'], df_train_clear1['rougher_recovery']))

#Сравнение концентрации материалов на каждом этапе очистки
print("------------------------------------------------------")
print("Средняя концентрация свинца после флотации", df_full["rougher.output.concentrate_pb"].mean())
print("Средняя концентрация свинца после первичного этапа очистки", df_full['primary_cleaner.output.concentrate_pb'].mean())
print("Средняя концентрация свинца на выходе", df_full["final.output.concentrate_pb"].mean())
print("------------------------------------------------------")
print("Средняя концентрация серебра после флотации", df_full["rougher.output.concentrate_ag"].mean())
print("Средняя концентрация серебра после первичного этапа очистки", df_full['primary_cleaner.output.concentrate_ag'].mean())
print("Средняя концентрация серебра на выходе", df_full["final.output.concentrate_ag"].mean())
print("------------------------------------------------------")
print("Средняя концентрация золота после флотации", df_full["rougher.output.concentrate_au"].mean())
print("Средняя концентрация золота после первичного этапа очистки", df_full['primary_cleaner.output.concentrate_au'].mean())
print("Средняя концентрация золота на выходе", df_full["final.output.concentrate_au"].mean())

#Сравнение распределений размеров гранул сырья
plt.title("Сравнение распределений размеров гранул сырья золота")
sns.set(style="darkgrid")
fig = sns.kdeplot(df_train["rougher.output.concentrate_au"], shade=True, color="r")
fig = sns.kdeplot(df_train["primary_cleaner.output.concentrate_au"], shade=True, color="b")
fig = sns.kdeplot(df_train["final.output.concentrate_au"], shade=True, color="g")
plt.legend(['Флотация', 'Первичная очистка', "На выходе" ], loc=1)
plt.show()

plt.title("Сравнение распределений размеров гранул сырья серебра")
sns.set(style="darkgrid")
fig = sns.kdeplot(df_train["rougher.output.concentrate_ag"], shade=True, color="r")
fig = sns.kdeplot(df_train["primary_cleaner.output.concentrate_ag"], shade=True, color="b")
fig = sns.kdeplot(df_train["final.output.concentrate_ag"], shade=True, color="g")
plt.legend(['Флотация', 'Первичная очистка', "На выходе" ], loc=1)
plt.show()

plt.title("Сравнение распределений размеров гранул сырья свинца")
sns.set(style="darkgrid")
fig = sns.kdeplot(df_train["rougher.output.concentrate_pb"], shade=True, color="r")
fig = sns.kdeplot(df_train["primary_cleaner.output.concentrate_pb"], shade=True, color="b")
fig = sns.kdeplot(df_train["final.output.concentrate_pb"], shade=True, color="g")
plt.legend(['Флотация', 'Первичная очистка', "На выходе" ], loc=1)
plt.show()

plt.title("Сравнение распределений размеров гранул сырья на выборках")
sns.set(style="darkgrid")
fig = sns.kdeplot(df_train['rougher.input.feed_size'], shade=True, color="r")
fig = sns.kdeplot(df_test['rougher.input.feed_size'], shade=True, color="b")
plt.legend(['Обучающая выборка', 'Тестовая выборка'], loc=1)
plt.show()

#Сравнение концентраций металлов на разных этапах обработки
df_train_clear1['input']=df_train_clear1['rougher.input.feed_ag']+df_train_clear1['rougher.input.feed_pb']+df_train_clear1['rougher.input.feed_au']\
                         +df_train_clear1['rougher.input.feed_sol']
df_train_clear1['rougher']=df_train_clear1['rougher.output.concentrate_ag']+df_train_clear1['rougher.output.concentrate_pb']+df_train_clear1['rougher.output.concentrate_au']\
                           +df_train_clear1['rougher.output.concentrate_sol']
df_train_clear1['black']=df_train_clear1['primary_cleaner.output.concentrate_ag']+df_train_clear1['primary_cleaner.output.concentrate_pb']\
                         +df_train_clear1['primary_cleaner.output.concentrate_au']+df_train_clear1['primary_cleaner.output.concentrate_sol']
df_train_clear1['finale']=df_train_clear1['final.output.concentrate_ag']+df_train_clear1['final.output.concentrate_pb']+df_train_clear1['final.output.concentrate_au']\
                          +df_train_clear1['final.output.concentrate_sol']
#Графическое представление полученных результатов
fig = plt.subplots(figsize=(20, 7))
fig = sns.distplot(a=df_train_clear1['rougher'], hist=True, kde=True, rug=False , color="r")
fig = sns.distplot(a=df_train_clear1['black'], hist=True, kde=True, rug=False , color="g")
fig = sns.distplot(a=df_train_clear1['finale'], hist=True, kde=True, rug=False , color="b")
fig = sns.distplot(a=df_train_clear1['input'], hist=True, kde=True, rug=False , color="y")
plt.legend(['rougher', 'primary', 'final', 'input'], loc=1)
plt.show()
#Отброс выброса
df_train_clear1=df_train_clear1.query('rougher > 20 & black>20 & finale>20 & input >20')
df_train_clear1['final_recovery']=df_train_clear1['final.output.recovery']
df_train_clear1=df_train_clear1.query('50 < final_recovery < 95')
print(df_train_clear1.shape)

#Статистические показатели обучающей выборки
df_train_clear1.boxplot('final.output.recovery')
df_train_clear1['final.output.recovery'].describe()

#ML
#Задание кастомной метрики
def smape(A, F):
    return 100/len(A) * np.sum(2 * np.abs(F - A) / (np.abs(A) + np.abs(F)))

total_smape=0.25*smape_rougher+0.75*smape_final1
print('Итоговое симметричное среднее абсолютное процентное отклонение', total_smape)

#Подготовка фичей и таргетов выборок
train_target1=df_train_clear1['rougher.output.recovery']
train_target2=df_train_clear1['final_recovery']
train_features=df_train_clear1[df_test.columns].drop('date', axis=1)
print(train_features.shape)

df_full_target=df_full[['date','rougher.output.recovery','final.output.recovery']]
df_test_ready = df_test.merge(df_full_target,  on='date', how='left')
df_test_ready=df_test_ready.dropna()

test_features=df_test_ready.drop(['date','rougher.output.recovery', 'final.output.recovery'], axis=1)
test_target1=df_test_ready['rougher.output.recovery']
test_target2=df_test_ready['final.output.recovery']
print(test_features.shape)

#Обучения моделей
#Линейная регрессия
model_lr =LinearRegression()
model_lr.fit(train_features, train_target1)
result=model_lr.predict(test_features)
smape1=smape(test_target1, result)
print(smape1)

model_lr1 =LinearRegression()
model_lr1.fit(train_features, train_target2)
result1=model_lr.predict(test_features)
smape2=smape(test_target2, result1)
print(smape2)

#Случайный лес (подбор параметров вставлять не стал) с кросс-валидацией
model_rougher=RandomForestRegressor(random_state=12345, n_estimators=40, max_depth=1)
model_rougher.fit(train_features, train_target1)
result=model_rougher.predict(test_features)
smape_rougher=smape(test_target1, result)
scores_rougher = cross_val_score(model_rougher, test_features, result, cv=5)
final_score = scores_rougher.mean()
print(smape_rougher)
print(scores_rougher)

model_final1=RandomForestRegressor(random_state=12345, n_estimators=40, max_depth=7)
model_final1.fit(train_features, train_target2)
result=model_final1.predict(test_features)
smape_final1=smape(test_target2, result)
scores_final = cross_val_score(model_final1, test_features, result, cv=5)
final_score1 = scores_final.mean()
print(smape_final1)
print(final_score1)

#Инициализация подбора гиперпараметров линейной регрессии
scores = make_scorer(smape, greater_is_better = False)
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


def lr_gridsearchcv(features, target, scores):
    model = LinearRegression()
    param_grid = {
        'copy_X': [True, False],
        'fit_intercept': [True, False],
        'normalize': [True, False]
    }
    my_scorer = make_scorer(scores, greater_is_better=False)
    CV = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring=my_scorer)
    CV.fit(features, target)
    print('Лучшее значение метрики: {:.2f}'.format(-CV.best_score_))
    print('Лучшие гиперпараметры: ', CV.best_params_)
    return CV.best_params_
lr_gridsearchcv(train_features, train_target1, smape)
lr_gridsearchcv(train_features, train_target2, smape)

#Сравнение результатов линейной регрессии с подобранными гиперпараметрами (которыепо сути дефолтные, не зря учил)
model_final=LinearRegression(copy_X=True, normalize = True)
model_final.fit(train_features, train_target2)
result1=model_final.predict(test_features)
smape_final=smape(test_target2, result1)
scores_final = cross_val_score(model_final, test_features, result1, cv=5)
final_score1 = scores_final.mean()
print(smape_final)


#Сравнение полученных результатов с дамми
from sklearn.dummy import DummyRegressor
dummy_regressor_rougher = DummyRegressor(strategy="median")
dummy_regressor_rougher.fit(train_features, train_target2)
dummy_rougher_pred = dummy_regressor_rougher.predict(test_features)
smape_dummy_rougher = smape(test_target2, dummy_rougher_pred)

print(smape_dummy_rougher)

dummy_regressor_final = DummyRegressor(strategy="median")
dummy_regressor_final.fit(train_features, train_target1)
dummy_final_pred = dummy_regressor_final.predict(test_features)
smape_dummy_final = smape(test_target1, dummy_final_pred)

print(smape_dummy_final)

total_smape=0.25*smape_dummy_rougher+0.75*smape_dummy_final
print(total_smape)