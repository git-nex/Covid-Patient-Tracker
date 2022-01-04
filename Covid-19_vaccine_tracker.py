import numpy as np
from numpy.core.fromnumeric import reshape
import pandas as pd
import matplotlib.pyplot as plt

file_1 = np.loadtxt('country_vac.csv', delimiter=',',
                    skiprows=1, dtype=np.string_)
# file_1 = np.genfromtxt('country_vac.csv', delimiter=',', skiprows=1, dtype=np.string_)
# print(file_1[0:,0:])
file_2 = np.array(file_1[0:, 2:])
# print(type(file_2))
file_2 = file_2.astype('int64')
# print(file_2)
# print(file_2[0:223].sum(axis=0))
Afgan = file_2[0:223].sum(axis=0)
Can = file_2[223:535].sum(axis=0)
Ind = file_2[535:816].sum(axis=0)
Unit = file_2[816:1123].sum(axis=0)
Rus = file_2[1123:].sum(axis=0)
# print(Afgan)
# print(Can)
# print(Ind)
# print(Unit)
# print(Rus)
arr = np.array([Afgan, Can, Ind, Unit, Rus])
# print()
# print(arr)
population = np.array(
    [[39864082], [3806790300], [139340903800], [33291507300], [14591202500]])

population.reshape(5, 1)

# print(population)
arr = np.concatenate((arr, population), axis=1)
# print(arr)

partially_vaccinated_Afg = (arr[0, 1]/arr[0, 3])*100
partially_vaccinated_Can = (arr[1, 1]/arr[1, 3])*100
partially_vaccinated_Ind = (arr[2, 1]/arr[2, 3])*100
partially_vaccinated_Unit = (arr[3, 1]/arr[3, 3])*100
partially_vaccinated_Rus = (arr[4, 1]/arr[4, 3])*100
# print(arr[0,3])
# print(partially_vaccinated_Afg)
# print(partially_vaccinated_Can)
# print(partially_vaccinated_Ind)
# print(partially_vaccinated_Unit)
# print(partially_vaccinated_Rus)
partially_vaccinated = np.array([[partially_vaccinated_Afg], [partially_vaccinated_Can], [partially_vaccinated_Ind],
                                 [partially_vaccinated_Unit], [partially_vaccinated_Rus]])


partially_vaccinated.reshape(5, 1)

partially_vaccinated = partially_vaccinated.astype('int64')
# print(partially_vaccinated)
# arr = np.concatenate((arr, partially_vaccinated), axis=1)
# print(arr)
completely_vaccinated_Afg = (arr[0, 2]/arr[0, 3])*100
completely_vaccinated_Can = (arr[1, 2]/arr[1, 3])*100
completely_vaccinated_Ind = (arr[2, 2]/arr[2, 3])*100
completely_vaccinated_Unit = (arr[3, 2]/arr[3, 3])*100
completely_vaccinated_Rus = (arr[4, 2]/arr[4, 3])*100
completely_vaccinated = np.array([[completely_vaccinated_Afg], [completely_vaccinated_Can], [completely_vaccinated_Ind],
                                  [completely_vaccinated_Unit], [completely_vaccinated_Rus]])

completely_vaccinated.reshape(5, 1)

# arr = np.concatenate((arr, completely_vaccinated), axis=1)
##################################################################################################################################
print("Country   |  Total Vaccinated | Partially Vaccinated | Completely Vaccinated | Total Population")
print("Afganistan" ,'   ',  arr[0,0],'           ',  arr[0,1],'           ',  arr[0,2],'               ',  arr[0,3])
print("Canada" ,'       ',  arr[1,0],'         ',  arr[1,1],'         ',  arr[1,2],'            '  ,  arr[1,3])
print("India" ,'        ',  arr[2,0],'        ',  arr[2,1],'        ',  arr[2,2],'            '  , arr[2,3])
print("USA" ,'          ',  arr[3,0],'        ',  arr[3,1],'        ',  arr[3,2],'           '  , arr[3,3])
print("Russia" ,'       ',  arr[4,0],'         ',  arr[4,1],'         ',  arr[4,2],'            '  ,  arr[4,3])

#########################################################################################################################################


writer = pd.ExcelWriter('demo.xlsx', engine='xlsxwriter')
writer.save()
# countryname=input("Give countryname")
df = pd.DataFrame({'Country': ['Afganistan', 'Canada', 'India', 'USA', 'Russia'],
                   'Total Vaccinated': [arr[0][0], arr[1][0], arr[2][0], arr[3][0], arr[4][0]],
                   'Partially Vaccinated': [(arr[0][1]), arr[1][1], arr[2][1], arr[3][1], arr[4][1]],
                   'Completely Vaccinated': [arr[0][2], arr[1][2], arr[2][2], arr[3][2], arr[4][2]],
                   'Total Population': [arr[0][3], arr[1][3], arr[2][3], arr[3][3], arr[4][3]]
                #    'Total Population': [arr[0][4], arr[1][4], arr[2][4], arr[3][4], arr[4][4]]
                   })


# Create a Pandas Excel writer using XlsxWriter as the engine.
writer = pd.ExcelWriter('demo.xlsx', engine='xlsxwriter')

# Convert the dataframe to an XlsxWriter Excel object.
df.to_excel(writer, sheet_name='Sheet1', index=False)

# Close the Pandas Excel writer and output the Excel file.
for column in df:
    column_width = max(df[column].astype(str).map(len).max(), len(column))
    col_idx = df.columns.get_loc(column)
    writer.sheets['Sheet1'].set_column(col_idx, col_idx, column_width)

writer.save()
# df.to_csv('myfile.csv')

###################################################################################################################################

# fig1
import mpl_toolkits.axes_grid1 as axes_grid1

plt.plot(arr)
plt.xlabel('months')
plt.ylabel('vaccination in million')
plt.title('vaccination status graph')
labels=['dec(2019)','jan-mar','apr-jun','jul-sep','oct-dec']
plt.xticks(range(len(labels)), labels, size='small')

# fig2
fig1, (ax1, ax2)= plt.subplots(2, sharex = True, sharey = False)
ax1.imshow(arr, interpolation ='none', aspect = 'auto')
ax2.imshow(arr, interpolation ='bicubic', aspect = 'auto')
for (j,i),label in np.ndenumerate(arr):
    ax1.text(i,j,label,ha='center',va='center')
    ax2.text(i,j,label,ha='center',va='center')

# fig3
fig = plt.figure()
grid = axes_grid1.AxesGrid(
    fig, 111, nrows_ncols=(1, 1), axes_pad = 0.5, cbar_location = "right",
    cbar_mode="each", cbar_size="15%", cbar_pad="5%",)
im1 = grid[0].imshow(arr, cmap='jet', interpolation='nearest')
grid.cbar_axes[0].colorbar(im1)

plt.show()