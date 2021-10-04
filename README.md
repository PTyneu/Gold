Целью данного проекта являлось построение модели, предсказывающей коэффициент восстановления золота из золотосодержащей руды, в качестве метрики качества былa взятa cимметричная средняя абсолютная ошибка в процентах (smape). В ходе работы была произведена предобработка данных (заполнение пропусков, оценка распределений и тд)  проведен исследовательский анализ данных(удаление выбросов, оценка распределений и их графическое представление) а также модельная часть (создание непосредственно метрики, обучение простых моделей: случайный лес, линейная регрессия, подбор гиперпараметров под кастомную метрику) а также сделаны некоторые аналитические выводы. 

В файле с расширением .py находится непосредственно скрипт данного проекта а также комментарии к каждому шагу, а в .ipynb образец полученных результатов и некоторые выводы по ним. Использованные библиотеки указаны в файле requirements.txt.

Полученные метрики удовлетворяли требованиям, что замечательно, тк не хотелось на таком объеме данных обучать более сложные модели. Проект можно улучшить проведя хотя бы какой-то фич инженеринг, оценить корреляции фичей с таргетом и отбросив совсем уж лишнее ну и разумеется подбор гиперпараметров это наше все
