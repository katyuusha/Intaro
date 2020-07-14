import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
import scipy as scipy
import scipy.stats as st
from scipy import stats
from scipy.stats import norm
import statistics as statistics
import random 
import numpy.random
from random import random, randrange, randint


data =  pd.read_csv("Speed-Filenames6.csv", delimiter=',')
np_speed = data.loc[:, 'Speed']
np_speed = np.sort(np_speed)
np_speed_once = np.array(np.unique(np_speed))

#1 интервальный ряд

n = len(np_speed)
x_min = np_speed.min()
x_max = np_speed.max()
h = round((x_max - x_min)/(1+3.322*math.log10(n)), 1) #огругленные длины частичных интервалов
m = round(1+3.322*math.log10(n)) #округленное число интервалов 
x_start = round(x_min - h/2)
x_end = round(x_min + h*m)
#неравенство выполнилось

np_st = np.array([])
np_en = np.array([])
for i in range(0, m):
    if (i == 0):
        np_st = np.append(np_st, x_min - h/2)
        np_en = np.append(np_en, x_min - h/2 +h )
    else:
        np_st = np.append(np_st, np_en[i-1])
        np_en = np.append(np_en, np_st[i] + h)
if (np_st[-1] <= x_max and x_max < np_en[-1]):
    print(2)
else:
    np_st = np.append(np_st, np_en[-1])
    np_en = np.append(np_en, np_st[-1] + h)
    m = m + 1


np_ni = np.array([]) #частоты
for i in range(0, m):
    np_ni = np.append(np_ni, sum(np.logical_and(np_speed >= np_st[i], np_speed < np_en[i])))

np_pi = np_ni/n #частости

for i in range (0,len(np_st)):
    if (i == 0):
        print("%12s %13s %11s" % ('Интервалы', 'Частоты', 'Частости'))
        print("[%2.2f ; %2.2f) %7i %13.2f" % (np_st[0], np_en[0], np_ni[i], np_pi[0]))
    else:
        print("[%2.2f ; %2.2f) %7i %13.2f" % (np_st[i], np_en[i], np_ni[i], np_pi[i]))

#2 точечные характеристики
np_x = np.array([])
np_x = (np_en + np_st)/ 2

x_vyb = np.sum(np_x*np_pi) #выборочное среднее
x_vyb_np = np_speed.mean()
print('Выборочное среднее: ', round(x_vyb, 2), x_vyb_np)
D = round(np.sum(((np_x - x_vyb)**2)*np_pi), 2) #дисперсия
D_np = round(np.var(np_speed), 2)
print('Дисперсия: ', D, D_np)
sigma = math.sqrt(D) #выборочное среднее квадратическое отклонение СКО
sigma_np = round(np.std(np_speed, ddof=1), 2)
print('СКО: ', round(sigma, 2), sigma_np)
S2 = round(D*(n/(n-1)), 2) # исправленная дисперсия
#S2_np = np.std(np_speed, ddof = n )
print('Исправленная дисперсия: ', S2)
mu_3 = np.sum(((np_x - x_vyb)**3)*np_pi) #центральный выборочный момент
A_s = round(mu_3/((sigma)**3), 2) # выборочный коэффициент ассиметрии
A_s_np = round(stats.skew(np_speed), 2)
print('Выборочный коэффициент ассиметрии: ', A_s, A_s_np)
mu_4 = np.sum(((np_x - x_vyb)**4)*np_pi) #центральный выборочный момент
E_k = round((mu_4/((sigma)**4)) - 3, 2)
E_k_np = round(stats.kurtosis(np_speed), 2)# выборочный коэффициент эксцесса
print('Выборочный коэффициент эксцесса: ', E_k, E_k_np)
M_0 = stats.mode(np_speed)#мода
print('Мода: ', M_0[0])
Me = (np_speed[49] + np_speed[50])/2#медиана
Me_np = np.median(np_speed)
print('Медиана: ', Me, Me_np)


#3 интервальные характеристики
#выборочное среднее
gamma = 0.95 #надежность оценки
Ft = 0 #функция Лапласа
2*Ft == gamma
Ft = gamma/2 
t = 1.96 #по таблице значений функции Лапласа
delta = round((t*sigma)/(math.sqrt(n)), 2) #нашли дельту
print('Доверительный интервал для среднего выборочного')
print('[', x_vyb - delta, ';', x_vyb + delta,']')

#СКО
tj = 1.98 #по таблице квантилей распределения Стьюдента
delta2 = round((sigma*tj)/(math.sqrt(n)), 2)
print('Доверительный интервал для СКО')
print('[',round(sigma - delta2, 2), ';', round(sigma + delta2,2) ,']')

#4 графики

#гистограмма

np_interval = np.array([])
np_interval = np.append(np_st, np_en[-1])
plt.bar(np_st, np_pi/h, h,  color = "grey", align='edge') #гистограмма
plt.xlabel('x')
plt.ylabel('pi/h')
tick_val = np_interval
tick_lab = ['57.3', '60.7', '64.1', '67.5', '70.9', '74.3', '77.7', '81.1', '84.5', '87.9']
plt.xticks(tick_val, tick_lab)

#полигон

np_interval_median = np.array(np_st + h/2)
from scipy.interpolate import make_interp_spline, BSpline
xnew = np.linspace(np_interval_median.min(), np_interval_median.max(), 400)
spl = make_interp_spline(np_interval_median, np_pi/h, k=3)
power_smooth = spl(xnew)
plt.plot(xnew, power_smooth, linewidth = 3, color = 'yellow')


plt.plot(np_interval_median, np_pi/h, color = "black")
plt.show()



#распределения
np_pi_new = np.array([]) #создаем для оси у 
for i in range(0,len(np_pi)):
    if (i == 0):
        np_pi_new = np.append(np_pi_new, np_pi[0])
    else:
        np_pi_new = np.append(np_pi_new, np_pi_new[i-1] + np_pi[i])       
plt.scatter(np_st, np_pi_new, color = 'red')
tick_val = np_interval
tick_lab = ['57.3', '60.7', '64.1', '67.5', '70.9', '74.3', '77.7', '81.1', '84.5', '']
plt.xticks(tick_val, tick_lab)

np_st_new = np.append(np_st, 90)
np_pi_new = np.append(np_pi_new, 1) 
for i in range(0,len(np_st)):
    if (i==9):
        x1, y1 = [np_st_new[i], np_st_new[i]], [np_pi_new[i], np_pi_new[i]]  
    else:
        x1, y1 = [np_st_new[i], np_st_new[i+1]], [np_pi_new[i], np_pi_new[i]]
    plt.plot(x1, y1, color = 'black')
plt.show()
plt.clf()

#5 гипотеза
np_interval = np.array([])
np_interval = np.append(np_st, np_en[-1])
plt.bar(np_st, np_pi/h, h,  color = "red", align='edge') #гистограмма
plt.xlabel('x')
plt.ylabel('pi/h')
tick_val = np_interval
tick_lab = ['57.3', '60.7', '64.1', '67.5', '70.9', '74.3', '77.7', '81.1', '84.5', '87.9']
plt.xticks(tick_val, tick_lab)

fit = stats.norm.pdf(np_speed, np.mean(np_speed), np.std(np_speed))  #this is a fitting indeed

plt.plot(np_speed,fit,'-o') #функция плотности для предполагаемого распределения

plt.show() 

#6
sigma_As = math.sqrt((6*(n-1))/((n+1)*(n+3))) #средние квадратические отклонения
sigma_Ek = math.sqrt(((24*n)*(n-2)*(n-3))/((n-1)**2*(n+3)*(n+5)))
if (math.fabs(A_s) <= 2*sigma_As and math.fabs(E_k) <= 2*sigma_Ek):
    print("Условие критерия выполняется")
#гипотезу принимаем и проверяем по критерию Пирсона
np_zi = np.array((np_st - x_vyb)/sigma)#границы интервала
np_zi1 = z_i = np.array((np_en - x_vyb)/sigma)

np_F_zi = np.array([])
np_F_zi1 = np.array([])
for i in range(0, len(np_zi)):
    np_F_zi = np.append(np_F_zi, scipy.stats.norm.cdf(np_zi[i]) - 0.5 ) #по таблице Лапласа
    np_F_zi1 = np.append(np_F_zi1, scipy.stats.norm.cdf(np_zi1[i]) - 0.5 )
np_F_zi[0] = -0.5
np_F_zi1[-1] = 0.5

np_Pi = np_F_zi1 - np_F_zi

np_ni_hatch = np_Pi*n


#сравним эмпирические и теоретические частоты
np_ni_minus_hatch = np_ni - np_ni_hatch
np_ni_minus_hatch2 = np_ni_minus_hatch**2
np_ni2 = np_ni**2
if (round(np.sum((np_ni2/np_ni_hatch))-n, 4) == round((np.sum(np_ni_minus_hatch2/np_ni_hatch))), 4):
    print('Расчеты проведены верно')
hi2_nab = round(np.sum((np_ni2/np_ni_hatch))-n, 4)

#зададим альфа
aplha = 0.5
k = m - 3 #число степеней свободы
hi2_kr = 12.6 
if(hi2_nab < hi2_kr):
    print('Нет оснований опровергнуть гипотезу H_0 о нормальном распределении генеральной совокупности X')
    
#7 
#jackknife
import random

np_20 = np.random.choice(np_speed, 20, replace=False)
np_80 = np.random.choice(np_speed, 80, replace=False)
np_100 = np.random.choice(np_speed, 100, replace=False)

np_20_vyb = np.array([])
np_20_D = np.array([])
np_20_sigma = np.array([])
np_20_S2 = np.array([])
np_20_As = np.array([])
np_20_Ek = np.array([])
np_20_M0 = np.array([])
np_20_Me = np.array([])
np_20_1 = np_20

np_80_vyb = np.array([])
np_80_D = np.array([])
np_80_sigma = np.array([])
np_80_S2 = np.array([])
np_80_As = np.array([])
np_80_Ek = np.array([])
np_80_M0 = np.array([])
np_80_Me = np.array([])
np_80_1 = np_80

np_100_vyb = np.array([])
np_100_D = np.array([])
np_100_sigma = np.array([])
np_100_S2 = np.array([])
np_100_As = np.array([])
np_100_Ek = np.array([])
np_100_M0 = np.array([])
np_100_Me = np.array([])
np_100_1 = np_80
np_100_1 = np_100

for i in range(0, len(np_20)):
    np_20_1 = np.delete(np_20_1, i)
    np_20_vyb = np.append(np_20_vyb, np_20_1.mean())
    np_20_D = np.append(np_20_D, np.var(np_20_1))
    np_20_sigma = np.append(np_20_sigma, np.std(np_20_1, ddof=1))
    np_20_S2 = np.append(np_20_S2, np_20_D*len(np_20_1)/(len(np_20_1)-1))
    np_20_As = np.append(np_20_As, stats.skew(np_20_1))
    np_20_Ek = np.append(np_20_Ek, stats.kurtosis(np_20_1))
    np_20_M0 = np.append(np_20_M0, stats.mode(np_20_1)[0])
    np_20_Me = np.append(np_20_Me, np.median(np_20_1))
    np_20_1 = np_20
    
for i in range(0, len(np_80)):
    np_80_1 = np.delete(np_80_1, i)
    np_80_vyb = np.append(np_80_vyb, np_80_1.mean())
    np_80_D = np.append(np_80_D, np.var(np_80_1))
    np_80_sigma = np.append(np_80_sigma, np.std(np_80_1, ddof=1))
    np_80_S2 = np.append(np_80_S2, np_80_D*len(np_80_1)/(len(np_80_1)-1))
    np_80_As = np.append(np_80_As, stats.skew(np_80_1))
    np_80_Ek = np.append(np_80_Ek, stats.kurtosis(np_80_1))
    np_80_M0 = np.append(np_80_M0, stats.mode(np_80_1)[0])
    np_80_Me = np.append(np_80_Me, np.median(np_80_1))
    np_80_1 = np_80
    
for i in range(0, len(np_100)):
    np_100_1 = np.delete(np_100_1, i)
    np_100_vyb = np.append(np_100_vyb, np_100_1.mean())
    np_100_D = np.append(np_100_D, np.var(np_100_1))
    np_100_sigma = np.append(np_100_sigma, np.std(np_100_1, ddof=1))
    np_100_S2 = np.append(np_100_S2, np_100_D*len(np_100_1)/(len(np_100_1)-1))
    np_100_As = np.append(np_100_As, stats.skew(np_100_1))
    np_100_Ek = np.append(np_100_Ek, stats.kurtosis(np_100_1))
    np_100_M0 = np.append(np_100_M0, stats.mode(np_100_1)[0])
    np_100_Me = np.append(np_100_Me, np.median(np_100_1))
    np_100_1 = np_100
    

print('Точечные характеристики по методу джекнайф')
print('\nДля 20% выборки:')
print('Выборочное среднее: ', round(np_20_vyb.mean(), 2))
print('Дисперсия: ', round(np_20_D.mean(),2))
print('СКО: ', round(np_20_sigma.mean(), 2))
print('Исправленная дисперсия: ', round(np_20_S2.mean(), 2))
print('Выборочный коэффициент ассиметрии: ', round(np_20_As.mean(), 2))
print('Выборочный коэффициент эксцесса: ', round(np_20_Ek.mean(), 2))
print('Мода: ', round(np_20_M0.mean(), 2))
print('Медиана: ', round(np_20_Me.mean(), 2))

print('\nДля 80% выборки:')
print('Выборочное среднее: ', round(np_80_vyb.mean(), 2))
print('Дисперсия: ', round(np_80_D.mean(),2))
print('СКО: ', round(np_80_sigma.mean(), 2))
print('Исправленная дисперсия: ', round(np_80_S2.mean(), 2))
print('Выборочный коэффициент ассиметрии: ', round(np_80_As.mean(), 2))
print('Выборочный коэффициент эксцесса: ', round(np_80_Ek.mean(), 2))
print('Мода: ', round(np_80_M0.mean(), 2))
print('Медиана: ', round(np_80_Me.mean(), 2))

print('\nДля 100% выборки:')
print('Выборочное среднее: ', round(np_80_vyb.mean(), 2))
print('Дисперсия: ', round(np_80_D.mean(),2))
print('СКО: ', round(np_80_sigma.mean(), 2))
print('Исправленная дисперсия: ', round(np_80_S2.mean(), 2))
print('Выборочный коэффициент ассиметрии: ', round(np_80_As.mean(), 2))
print('Выборочный коэффициент эксцесса: ', round(np_80_Ek.mean(), 2))
print('Мода: ', round(np_80_M0.mean(), 2))
print('Медиана: ', round(np_80_Me.mean(), 2))

#bootstrap 
#выборки
a=20
b=80
c=100
np_20_0 = np.array([numpy.random.choice(np_speed, size=(1000,a), replace=True, p=None)])
np_20_0 = np_20_0[0]
np_80_0 = np.array([numpy.random.choice(np_speed, size=(1000, b), replace=True, p=None)])
np_80_0 = np_80_0[0]
np_100_0 = np.array([numpy.random.choice(np_speed, size=(1000, c), replace=True, p=None)])
np_100_0 = np_100_0[0]

np_20_vyb0 = np.array([])
np_20_D0 = np.array([])
np_20_sigma0 = np.array([])
np_20_S20 = np.array([])
np_20_As0 = np.array([])
np_20_Ek0 = np.array([])
np_20_M00 = np.array([])
np_20_Me0 = np.array([])

np_80_vyb0 = np.array([])
np_80_D0 = np.array([])
np_80_sigma0 = np.array([])
np_80_S20 = np.array([])
np_80_As0 = np.array([])
np_80_Ek0 = np.array([])
np_80_M00 = np.array([])
np_80_Me0 = np.array([])

np_100_vyb0 = np.array([])
np_100_D0 = np.array([])
np_100_sigma0 = np.array([])
np_100_S20 = np.array([])
np_100_As0 = np.array([])
np_100_Ek0 = np.array([])
np_100_M00 = np.array([])
np_100_Me0 = np.array([])

for i in range(0, 1000):
    np_20_vyb0 = np.append(np_20_vyb0, np_20_0[i].mean())
    np_20_D0 = np.append(np_20_D0, np.var(np_20_0[i]))
    np_20_sigma0 = np.append(np_20_sigma0, np.std(np_20_0[i], ddof=1))
    np_20_S20 = np.append(np_20_S20, np_20_D0*len(np_20_0[i])/(len(np_20_0[i])-1))
    np_20_As0 = np.append(np_20_As0, stats.skew(np_20_0[i]))
    np_20_Ek0 = np.append(np_20_Ek0, stats.kurtosis(np_20_0[i]))
    np_20_M00 = np.append(np_20_M00, stats.mode(np_20_0[i])[0])
    np_20_Me0 = np.append(np_20_Me0, np.median(np_20_0[i]))
    
for i in range(0, 1000):
    np_80_vyb0 = np.append(np_80_vyb0, np_80_0[i].mean())
    np_80_D0 = np.append(np_80_D0, np.var(np_80_0[i]))
    np_80_sigma0 = np.append(np_80_sigma0, np.std(np_80_0[i], ddof=1))
    np_80_S20 = np.append(np_80_S20, np_80_D0*len(np_80_0[i])/(len(np_80_0[i])-1))
    np_80_As0 = np.append(np_80_As0, stats.skew(np_80_0[i]))
    np_80_Ek0 = np.append(np_80_Ek0, stats.kurtosis(np_80_0[i]))
    np_80_M00 = np.append(np_80_M00, stats.mode(np_80_0[i])[0])
    np_80_Me0 = np.append(np_80_Me0, np.median(np_80_0[i]))
    
for i in range(0, 1000):
    np_100_vyb0 = np.append(np_100_vyb0, np_100_0[i].mean())
    np_100_D0 = np.append(np_100_D0, np.var(np_100_0[i]))
    np_100_sigma0 = np.append(np_100_sigma0, np.std(np_100_0[i], ddof=1))
    np_100_S20 = np.append(np_100_S20, np_100_D0*len(np_100_0[i])/(len(np_100_0[i])-1))
    np_100_As0 = np.append(np_100_As0, stats.skew(np_100_0[i]))
    np_100_Ek0 = np.append(np_100_Ek0, stats.kurtosis(np_100_0[i]))
    np_100_M00 = np.append(np_100_M00, stats.mode(np_100_0[i])[0])
    np_100_Me0 = np.append(np_100_Me0, np.median(np_100_0[i]))
    

print('\nТочечные характеристики по методу бутстреп')
print('\nДля 20% выборки:')
print('Выборочное среднее: ', round(np_20_vyb0.mean(), 2))
print('Дисперсия: ', round(np_20_D0.mean(),2))
print('СКО: ', round(np_20_sigma0.mean(), 2))
print('Исправленная дисперсия: ', round(np_20_S20.mean(), 2))
print('Выборочный коэффициент ассиметрии: ', round(np_20_As0.mean(), 2))
print('Выборочный коэффициент эксцесса: ', round(np_20_Ek0.mean(), 2))
print('Мода: ', round(np_20_M00.mean(), 2))
print('Медиана: ', round(np_20_Me0.mean(), 2))

print('\nДля 80% выборки:')
print('Выборочное среднее: ', round(np_80_vyb0.mean(), 2))
print('Дисперсия: ', round(np_80_D0.mean(),2))
print('СКО: ', round(np_80_sigma0.mean(), 2))
print('Исправленная дисперсия: ', round(np_80_S20.mean(), 2))
print('Выборочный коэффициент ассиметрии: ', round(np_80_As0.mean(), 2))
print('Выборочный коэффициент эксцесса: ', round(np_80_Ek0.mean(), 2))
print('Мода: ', round(np_80_M00.mean(), 2))
print('Медиана: ', round(np_80_Me0.mean(), 2))

print('\nДля 100% выборки:')
print('Выборочное среднее: ', round(np_80_vyb0.mean(), 2))
print('Дисперсия: ', round(np_80_D0.mean(),2))
print('СКО: ', round(np_80_sigma0.mean(), 2))
print('Исправленная дисперсия: ', round(np_80_S20.mean(), 2))
print('Выборочный коэффициент ассиметрии: ', round(np_80_As0.mean(), 2))
print('Выборочный коэффициент эксцесса: ', round(np_80_Ek0.mean(), 2))
print('Мода: ', round(np_80_M00.mean(), 2))
print('Медиана: ', round(np_80_Me0.mean(), 2))

#8 интервальные

#выборочное среднее
gamma = 0.95 #надежность оценки
Ft = 0 #функция Лапласа
2*Ft == gamma
Ft = gamma/2 
t = 1.96 #по таблице значений функции Лапласа


#выборочное среднее
delta_20 = (t*np_20_sigma.mean())/(math.sqrt(len(np_20)-1)) #нашли дельту
delta_80 = (t*np_80_sigma.mean())/(math.sqrt(len(np_80)-1))
delta_100 = (t*np_100_sigma.mean())/(math.sqrt(len(np_100)-1))
print('\nДоверительный интервал по методу джекнайф')
print('\nДля среднего выборочного')
print('Для 20% выборки')
print('[', round(np_20_vyb.mean() - delta_20,2), ';', round(np_20_vyb.mean() + delta_20,2),']')
print('Для 80% выборки')
print('[', round(np_80_vyb.mean() - delta_80,2), ';', round(np_80_vyb.mean() + delta_80,2),']')
print('Для 100% выборки')
print('[', round(np_100_vyb.mean() - delta_100,2), ';', round(np_100_vyb.mean() + delta_100,2),']')

#СКО
tj1 = 2.09
tj2 = 1.99
tj3 = 1.98 #по таблице квантилей распределения Стьюдента 
delta2_20 = (np_20_sigma.mean()*tj1)/(math.sqrt(len(np_20)-1))
delta2_80 = (np_80_sigma.mean()*tj2)/(math.sqrt(len(np_80)-1))
delta2_100 = (np_100_sigma.mean()*tj3)/(math.sqrt(len(np_100)-1))
                
print('\nДля СКО')
print('Для 20% выборки')
print('[',round(np_20_sigma.mean() - delta2_20, 2), ';', round(np_20_sigma.mean() + delta2_20,2) ,']')
print('Для 80% выборки')
print('[',round(np_80_sigma.mean() - delta2_80, 2), ';', round(np_80_sigma.mean() + delta2_80,2) ,']')
print('Для 100% выборки')
print('[',round(np_100_sigma.mean() - delta2_100, 2), ';', round(np_100_sigma.mean() + delta2_100,2) ,']')

#по бутстреп
#выборочное среднее
delta_20_0 = (t*np_20_sigma0.mean())/(math.sqrt(len(np_20_0[0]))) #нашли дельту
delta_80_0 = (t*np_80_sigma0.mean())/(math.sqrt(len(np_80_0[0])))
delta_100_0 = (t*np_100_sigma.mean())/(math.sqrt(len(np_100_0[0])))
print('\nДоверительный интервал по методу бутстреп')
print('\nДля среднего выборочного')
print('Для 20% выборки')
print('[', round(np_20_vyb0.mean() - delta_20_0,2), ';', round(np_20_vyb0.mean() + delta_20_0,2),']')
print('Для 80% выборки')
print('[', round(np_80_vyb0.mean() - delta_80_0,2), ';', round(np_80_vyb0.mean() + delta_80_0,2),']')
print('Для 100% выборки')
print('[', round(np_100_vyb0.mean() - delta_100_0,2), ';', round(np_100_vyb0.mean() + delta_100_0,2),']')

#СКО
delta2_20_0 = (np_20_sigma0.mean()*tj1)/(math.sqrt(len(np_20_0[0])))
delta2_80_0 = (np_80_sigma0.mean()*tj2)/(math.sqrt(len(np_80_0[0])))
delta2_100_0 = (np_100_sigma0.mean()*tj3)/(math.sqrt(len(np_100_0[0])))
                
print('\nДля СКО')
print('Для 20% выборки')
print('[',round(np_20_sigma0.mean() - delta2_20, 2), ';', round(np_20_sigma0.mean() + delta2_20_0,2) ,']')
print('Для 80% выборки')
print('[',round(np_80_sigma0.mean() - delta2_80, 2), ';', round(np_80_sigma0.mean() + delta2_80_0,2) ,']')
print('Для 100% выборки')
print('[',round(np_100_sigma0.mean() - delta2_100, 2), ';', round(np_100_sigma0.mean() + delta2_100_0,2) ,']')

#графически
#среднее выборочное
print('\nГрафическое представление')
x = np.array([x_vyb, np_20_vyb.mean(), np_80_vyb.mean(), np_100_vyb.mean(), np_20_vyb0.mean(), np_80_vyb0.mean(), np_100_vyb0.mean()])
y = np.array([1,2,3,4,5,6,7])
err = np.array([delta, delta_20, delta_80, delta_100, delta_20_0, delta_80_0, delta_100_0])

#err = [random.uniform(0,1) for i in range(10)]
for i in range(0,7):
    plt.errorbar(x[i], y[i], xerr=err[i], 
        fmt='ko',marker='o',
       color='k',
       ecolor='k',
       markerfacecolor='r',
       label="series 2",
       capsize=4,
       linestyle='None')
data_names = ['', 'ст.метод', 'JN 20%', 'JN 80%', 'JN 100%','B 20%','B 80%', 'B 100%']
xs = range(len(data_names))
plt.yticks(xs, data_names, rotation = 8)
plt.show()

#СКО
plt.clf()
x1 = np.array([sigma, np_20_sigma.mean(), np_80_sigma.mean(), np_100_sigma.mean(), np_20_sigma.mean(), np_80_sigma.mean(), np_100_sigma.mean()])
y1 = np.array([1,2,3,4,5,6,7])
err1 = np.array([delta2, delta2_20, delta2_80, delta2_100, delta2_20_0, delta2_80_0, delta2_100_0])

#err = [random.uniform(0,1) for i in range(10)]
for i in range(0,7):
    plt.errorbar(x1[i], y1[i], xerr=err1[i], 
        fmt='ko',marker='o',
       color='k',
       ecolor='k',
       markerfacecolor='r',
       label="series 2",
       capsize=4,
       linestyle='None')
data_names1 = ['', 'ст.метод', 'JN 20%', 'JN 80%', 'JN 100%','B 20%','B 80%', 'B 100%']
xs1 = range(len(data_names1))
plt.yticks(xs1, data_names1, rotation = 8)
plt.show()


#10
print('\nПредположим, что у нас нормальное распределение')

#11
tj = 1.98 #по таблице квантилей распределения Стьюдента
a = np_100_vyb0.mean()
b = np_100_sigma0.mean()
stud_boot = np.array([])
stud_start = np.array([])
x_vyb_stud = np.array([])
for i in range(0,1000):
    if i == 0:
        new = np.array([np.random.normal(a, b, size=100)])
    else:
        k = np.array([np.random.normal(a, b, size=100)])
        new = np.vstack([new, k])
    x_vyb_stud = np.append(x_vyb_stud, new[i].mean())
    
for i in range(0,1000):
    stud_boot = np.append(stud_boot, stats.ttest_1samp(new[i],np_100_vyb0.mean())[0]) 

stud_start = stats.ttest_1samp(np_speed,x_vyb)[0] 

tj_boot = 1.96
x_boot = (np.sum(stud_boot))/1000
se_boot = ((np.sum((stud_boot - x_boot)**2))/999)**(1/2)


int_start = x_boot - tj_boot*se_boot
int_end = x_boot + tj_boot*se_boot

if(stud_start >= int_start and stud_start <= int_end):
    print('\nПопадание в интервал [', round(int_start, 2), ';', round(int_end, 2), '].Гипотезу о нормальном распределении принимаем.')
else:
    print('\nНепопадание в интервал [', round(int_start, 2), ';', round(int_end, 2), ']. Гипотезу о нормальном распределении опровергаем.')