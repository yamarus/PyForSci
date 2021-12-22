.. _theme11:

==============================
Полезные библиотеки для python
==============================

`Лекция в .ipynb формате <../../source/lectures/theme11.ipynb>`_

Pandas
======

Pandas - открытая библиотека, предоставляющая высокпроизводительные
удобные инструменты для работы с данными. Библиотека поставляется в
стандартном наборе дистрибутива anaconda. `Ссылка на документацию <https://pandas.pydata.org/docs/index.html>`_.

Для использования ``Pandas`` импортируется в код обычным методом. По
общей договоренности чаще всего его импортируют как ``pd``.

`Чуть более подробное описание библиотеки <theme10.html>`_

.. code:: ipython3

    import pandas as pd

Один пример работы Pandas. Загрузка экспериментальных данных
------------------------------------------------------------

Для загрузки экспериментальных данных из текстовых файлов используется
функция ``pd.read_table(filename)``

В нашем примере мы создадим функцию ``importer`` с аргументами
``folder`` (папка с экспериментальными данными) и ``undent`` (число,
показывающее сколько символов с конца необходимо убрать из имени файла,
чтобы получить число; в дальнейшем это будет использоваться для задания
ключей).

.. code:: ipython3

    def importer(folder, undent):    
        items = os.listdir(folder)#Считываем имена файлов в папке
        data = pd.DataFrame()#Создаем пустой DataFrame
        
        for names in items: #Инициируем главный цикл перебора всех имен в искомой папке
            
            if names.endswith(".txt"):#Выбираем только файлы с расширением txt. 
                
                #Загружаем данные из файлов текстовых файлов
                table=pd.read_table(folder +'/' + names, # имя файла
                                    sep='\s+', #разделитель строк в исходном файле
                                    skiprows=[0],#количество или номера строк, которые не следует считывать 
                                    names=['wavelenght'+names[:-undent],int(names[:-undent])], #названия колонок, которые следует использовать при чтении из файла
                                    index_col=0)#колонка, которая будет использоваться в качестве индекса. Если не задавать - будут присвоены индексы по умолчанию, и датафрейм будет двумерным
                #Собираем все экспериментальные в один датафрейм data с помощью конкатенации массивов data и table.
                #В отличие от numpy при конкатенации dataframe нулевой размерности и непустого не возникает ошибок
                #несовпадения размерности, что значитльно упрощает жизнь. Кроме этого, как было указано раньше,
                #при несовпадении размерностей файлов, пустые ячейки будут заполнены NaN, после чего их можно будет убрать, например взяв срез
                data = pd.concat([data,table], 
                                 axis=1)#ключ, указывающий на поколоночную конкатенацию  
        return data.reindex(sorted(data.columns), axis=1)

На следующем этапе воспользуемся созданной нами функцией importer.

`используемые файлы <../../source/lectures/theme10.zip>`_

.. code:: ipython3

    import os
    folder="theme10/"#задаем название папки с экспериментальными данными
    undent = 7 #количесвто ненужных символов в конце названия
    data = importer(folder,undent) #задаем переменной data значения функции importer c аргументами folder и udent
    print(data)


.. parsed-literal::

            -2     -1      0      10     20     30     40     50     60     70   \
    700.0  0.010  0.010  0.012  0.001  0.002  0.005  0.001  0.002  0.002  0.002   
    699.0  0.010  0.010  0.012  0.001  0.002  0.005  0.001  0.002  0.002  0.002   
    698.0  0.010  0.010  0.012  0.001  0.002  0.005  0.001  0.002  0.002  0.002   
    697.0  0.009  0.010  0.012  0.001  0.002  0.004  0.001  0.002  0.001  0.002   
    696.0  0.009  0.010  0.012  0.001  0.002  0.005  0.001  0.002  0.001  0.002   
    ...      ...    ...    ...    ...    ...    ...    ...    ...    ...    ...   
    204.0  0.147  0.127  0.108 -0.013  0.006  0.036  0.012  0.085  0.045  0.002   
    203.0  0.240  0.068  0.072  0.005  0.025  0.062  0.039  0.036  0.084  0.017   
    202.0  0.121  0.017 -0.009 -0.025 -0.064  0.008 -0.005  0.024  0.036 -0.011   
    201.0  0.077  0.115  0.059 -0.030  0.004  0.049 -0.002 -0.028  0.014 -0.042   
    200.0  0.087  0.125  0.027 -0.032 -0.050  0.067  0.035  0.061  0.025 -0.021   
    
             80     90     100    110    120    130    140    150    160    170  
    700.0  0.001  0.003  0.004  0.002  0.001  0.006  0.004  0.002  0.003  0.001  
    699.0  0.001  0.003  0.004  0.002  0.001  0.006  0.004  0.002  0.003  0.001  
    698.0  0.001  0.003  0.004  0.002  0.001  0.006  0.004  0.002  0.003  0.001  
    697.0  0.001  0.003  0.004  0.002  0.001  0.006  0.004  0.002  0.003  0.001  
    696.0  0.001  0.003  0.004  0.002  0.001  0.006  0.004  0.002  0.003  0.001  
    ...      ...    ...    ...    ...    ...    ...    ...    ...    ...    ...  
    204.0  0.019  0.001  0.065  0.005  0.074  0.005  0.095  0.316  0.343  0.499  
    203.0  0.019  0.142  0.076  0.070  0.091  0.082  0.071  0.482  0.504  0.413  
    202.0  0.025 -0.001  0.022 -0.023  0.037  0.034  0.068  0.369  0.456  0.321  
    201.0 -0.001  0.014  0.036  0.011  0.037 -0.029  0.046  0.352  0.452  0.439  
    200.0 -0.003  0.014 -0.027 -0.003  0.020  0.071  0.083  0.520  0.367  0.462  
    
    [501 rows x 20 columns]


Построим сперва график одной экспериментальной кривой на первом графике,
а затем на втором - всех экспериментальных данных.

.. code:: ipython3

    data[{0}].plot()
    data.plot() 




.. parsed-literal::

    <AxesSubplot:>




.. image:: figs/theme11/output_7_1.png



.. image:: figs/theme11/output_7_2.png


Теперь вычтем исходны спектр из остальных, и построим изменение сигнала
на 300 нм от времени

.. code:: ipython3

    first_col=data[{0}]
    
    bigtable_sub=data.apply(lambda column: column - first_col[0],  
                            #column взято для удобства, можно писать любое название
                            #first_col[0] необходимо повторно указать название колонки,
                            #потому что pandas требует это делать при применении оперпций к колонкам
                            axis=0)# ключ указывает, что опреацию будем применять поколоночно
    
    bigtable_sub.plot()





.. parsed-literal::

    <AxesSubplot:>




.. image:: figs/theme11/output_9_1.png


.. code:: ipython3

    data300 = data.loc[300]
    ax=data300.plot()
    ax.set_xlabel("Time, min") #название оси x
    ax.set_ylabel("Absorption")   #название оси y




.. parsed-literal::

    Text(0, 0.5, 'Absorption')




.. image:: figs/theme11/output_10_1.png


Seaborn
=======

Seaborn - библиотека для визуализации основаная на ``matplotlib``.
Основным направлением библиотеки является работа со статистическими
данными.

`Документация Seaborn <https://seaborn.pydata.org/index.html>`_

Расмотрим пример распределения глубины клюва пингвинов в зависимости от
массы тела

.. code:: ipython3

    import seaborn as sns
    sns.set_theme(style="white")
    
    df = sns.load_dataset("penguins")
    #print(type(df),'\n\n',df)
    
    g = sns.JointGrid(data=df, x="body_mass_g", y="bill_depth_mm", space=0)
    g.plot_joint(sns.kdeplot,
                 fill=True, clip=((2200, 6800), (10, 25)),
                 thresh=0, levels=100, cmap="rocket")
    g.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=25)




.. parsed-literal::

    <seaborn.axisgrid.JointGrid at 0x7fa6ea294c40>




.. image:: figs/theme11/output_12_1.png

SymPy
=====

Пакет для символьных вычислений.

`Документация <https://www.sympy.org/en/index.html>`_

.. code:: ipython3

    import sympy
    from sympy import init_printing, symbols
    init_printing(use_unicode=True)
    x, y = symbols('x y')
    expr = (x + y)**2
    expr




.. math::

    \displaystyle \left(x + y\right)^{2}



.. code:: ipython3

    from sympy import expand, factor
    expanded_expr = expand(expr)
    expanded_expr




.. math::

    \displaystyle x^{2} + 2 x y + y^{2}



.. code:: ipython3

    factor(expanded_expr-x**2)




.. math::

    \displaystyle y \left(2 x + y\right)



.. code:: ipython3

    from sympy import integrate, exp,cos,sin
    integrate(exp(x)*sin(x) + exp(x)*cos(x), x)




.. math::

    \displaystyle e^{x} \sin{\left(x \right)}


symfit
======

Пакет, комбинирующий ``scipy.optimize`` и ``sympy`` позволяет быстро и
просто проводить фитирование кривых аналитическими функциями.
`Документация <https://symfit.readthedocs.io/en/stable/index.html>`_.

scikit-learn
============

Библиотека для машинного обучения. `Сайт проекта <https://scikit-learn.org/>`_. Собрана на основе
NumPy, SciPy, matplolib

svgwrite
========

Библиотека для отрисовки svg. `Документация <https://svgwrite.readthedocs.io/en/latest/>`_.

Awesome Python Chemistry
========================

Сообщество ученых, использующих Python составило и поддерживает
актуальным список полезных пакетов для химимков. Ниже представлены
некоторые программы из этого списка. Полный набор с кратким описанием
можно найти на `странице проекта <https://github.com/lmmentel/awesome-python-chemistry>`_

RDKit
=====

RDKit - мощный дистрибутив для хемоинформатики. Заточен для анализа
структур молекул. Требует установки дистрибутива.

`Сайт проекта <https://www.rdkit.org/>`_

.. code:: ipython3

    from rdkit import Chem

чтение данных может происходить из различных форматов:

.. code:: ipython3

    mol = Chem.MolFromSmiles('COc1cccc2cc(C(=O)NCCCCN3CCN(c4cccc5nccnc54)CC3)oc21')
    refmol = Chem.MolFromSmiles('CCCN(CCCCN1CCN(c2ccccc2OC)CC1)Cc1ccc2ccccc2c1')
    #mol = Chem.MolFromMolFile('data/input.mol')
    #stringWithMolData=open('data/input.mol','r').read()
    #mol = Chem.MolFromMolBlock(stringWithMolData)

Удобен для отрисовки молекул и визуализации данных.

.. code:: ipython3

    #from rdkit.Chem import AllChem
    from rdkit.Chem.Draw import IPythonConsole
    IPythonConsole.ipython_useSVG=True
    IPythonConsole.molSize = 400,200
    IPythonConsole.drawOptions.addStereoAnnotation = False

.. code:: ipython3

    mol




.. image:: figs/theme11/output_19_0.svg



.. code:: ipython3

    refmol




.. image:: figs/theme11/output_20_0.svg



.. code:: ipython3

    from rdkit.Chem.Draw import SimilarityMaps

.. code:: ipython3

    fig, maxweight = SimilarityMaps.GetSimilarityMapForFingerprint(refmol, mol, SimilarityMaps.GetMorganFingerprint)



.. image:: figs/theme11/output_22_0.png


datamol
-------

datamol - пакет надстройка над RDKit упрощает многие операкции при
работе с RDKit. `Документация <https://doc.datamol.io/>`_.

Open Babel
==========

Open Babel - инструмент для работы с химическими данными и переводом их
из одного формата в другой. Имеет ряд дополнительных инструментов для
анализа. `Документация <http://openbabel.org/wiki/Main_Page>`_.

Pybel
-----

Надстройак над Open Babel. Упрощает взаимодействие и дополнительно
адаптирует Open Babel для работы из python. `Документация <https://openbabel.org/docs/dev/UseTheLibrary/Python_Pybel.html>`_.

Pycroscopy
==========

Инструмент для работы с данными по микроскопии наноматериалов. `Сайт
проекта <https://pycroscopy.github.io/pycroscopy/index.html>`_.

stk
===

Инструмент для конструирования молекул. `Документация <https://stk.readthedocs.io/en/stable/index.html>`_.

nglviewer
=========

Виджет для IPython/Jupyter для визуализаций молекул и траекторий.
`Станица проекта <https://github.com/nglviewer/nglview>`_.

.. code:: ipython3

    import nglview
    view = nglview.show_file("1C4D.pdb")  
    #view = nglview.show_pdbid("3pqr") 
    view



.. parsed-literal::

    NGLWidget()


MDTraj
======

Библиотека для работы с траекториями молекулярной динамики. `Сайт <https://www.mdtraj.org/>`_
проекта.

python-docx
===========

Библиотека для работы с docx файлами. `Документация <https://python-docx.readthedocs.io/en/latest/>`_.

(c Excel лучше работать с помощью Pandas)

SPHINX
======

Генератор документации (например эта страница сделана с его помощью).
`Документация <https://www.sphinx-doc.org/en/master/>`_.
