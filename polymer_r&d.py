import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import pandas as pd
import random

#엑셀에서 Data 불러오기
dt = pd.read_excel('C:/Users/user/polymer_r&d_project_data.xlsx')

# shear rate column
x_columns = dt[['shear rate']]

# Viscosity value
y_columns = ['viscosity (460K)', 'viscosity (480K)', 'viscosity (500K)', 'viscosity (520K)', 'viscosity (540K)']

models = []
slope = []
log_zero_shear_viscosity = []

# K,Ta 구하기
for i in y_columns:
    y = dt[i][0:5]
    model = LinearRegression()
    model.fit(x_columns[:5], np.log(y))
    models.append(model)

    intercept = model.intercept_    #y절편
    slope_values = model.coef_[0]   #기울기
    slope.append(slope_values)
    log_zero_shear_viscosity.append(intercept)

#온도 역수
temperature = np.array([[1 / float(460)], [1 / float(480)], [1 / float(500)], [1 / float(520)], [1 / float(540)]])

model2 = LinearRegression()
model2.fit(temperature, log_zero_shear_viscosity)
y_pred2 = model2.predict(temperature)

shear_rate = dt['shear rate']

zero_shear_viscosity = []
zero_shear_viscosity = np.exp(log_zero_shear_viscosity)

# #검증
# plt.scatter(temperature, log_zero_shear_viscosity, color='blue', label='Data')
# plt.plot(temperature, y_pred2, color='red', label='Linear Regression')
# plt.xlabel('1/temperature(K)')
# plt.ylabel('shear rate values')
# plt.show()


rows = 5    #temperature
cols = 16   #shear rate
y2 = []     #viscosity

for i, column in enumerate(y_columns):
    ylist = dt[column][0:17]
    y2.append(ylist)

sseMin = float('inf')
repeat = 100
minMax = []
repeatTimes = 1

aValue = 0
aMax = 1
aMin = 0

bValue = 0
bMin = 0
bMax = 1

cMax = 10
cMin = 0
log_sigma_0Value = 0
log_sigma_0Min = 0
log_sigma_0Max = 10

a = 0
b = 0
log_sigma_0 = 0

#sse값이 15000이하로 떨어질 때 까지 반복시행
min_value = [1000000000, 0, 0, 0]
max_value = [100000000000, 0, 0, 0]
margin_of_error = 0.01

while max_value[0] - min_value[0] >= margin_of_error:
    for i in range(repeat):
        sseSum = 0
        a = random.uniform(aMin, aMax)  # 0 이상 1 미만의 랜덤값 생성
        b = random.uniform(bMin, bMax)  # 0 이상 1 미만의 랜덤값 생성
        log_sigma_0 = random.uniform(log_sigma_0Min, log_sigma_0Max)  # 1 이상 exp(10) 미만의 랜덤값 생성
        for j in range(rows):
            for k in range(cols):
                A = y2[j][k]
                B = zero_shear_viscosity[j]/np.power(1 + np.power((shear_rate * zero_shear_viscosity[j]) / np.exp(log_sigma_0), a), b)
                sseSum += np.power(A - B[k], 2)
        minMax.append([sseSum,a, b, log_sigma_0])
    for row in minMax:
        if row[0] < min_value[0]:
            min_value = row
        elif row[0] != min_value[0] and row[0] < max_value[0]:
            max_value = row
        else:
            continue
    aMax = max_value[1]
    bMax = max_value[2]
    log_sigma_0Max = max_value[3]
    aMin = min_value[1]
    bMin = min_value[2]
    log_sigma_0Min = min_value[3]
    aValue = (aMax + aMin) / 2
    bValue = (bMax + bMin) / 2
    log_sigma_0Value = (log_sigma_0Max + log_sigma_0Min) / 2
    
    # print(repeatTimes)
    # print(max_value)
    # print(min_value)

    # repeatTimes += 1
    # if repeatTimes >= 7 and min_value[0] > 50000:
    #     aMax = 1
    #     aMin = 0
    #     bMax = 1
    #     bMin = 0
    #     log_sigma_0Min = 0
    #     log_sigma_0Max = 10

    #     repeatTimes = 0
    #     minMax = []


#검정
print("repeatTimes: ",repeatTimes)


for i in range(rows):
    #기존 data
    plt.scatter(shear_rate, y2[i])
    #몬테카를로 모델을 통해 구한 파라미터  
    plt.loglog(shear_rate, zero_shear_viscosity[i]/(np.power(1 + np.power((shear_rate * zero_shear_viscosity[i]) / np.exp(log_sigma_0Value), aValue), bValue)))
plt.xlabel('shear_rate_value')
plt.ylabel('viscosity')
plt.show()

print("log T_a value: ", round(np.log(model2.coef_[0]),2))
print("log K value: ", round(model2.intercept_, 2))
print("log sigma_0 value: ", round(log_sigma_0Value, 2))
print("a value: ", round(aValue, 4))
print("b value: ", round(bValue, 4))