N  = 100        # Number of points
D  = 2          # Dimension
Px = 5          # Polynomial order in x-direction
Py = 5          # Polynomial order in y-direction

noise = normal(0,0.1,N)

x = uniform(0,1,N)
y = uniform(0,1,N)
z = FrankeFunction(x, y) + noise

order5 = Reg_2D(x, y, z, Px, Py)

beta_ols = order5.ols()
beta_ridge = order5.ridge()
beta_lasso = order5.lasso()

#plt.imshow(beta_ols)
#plt.show()

#plt.imshow(beta_ridge)
#plt.show()

#plt.imshow(beta_lasso)
#plt.show()


x_vals = y_vals = np.linspace(0,1,1000)
X_vals, Y_vals = np.meshgrid(x_vals, y_vals)
predict = polyval(X_vals, Y_vals, beta_ols)

# PLOT
fig = plt.figure() 
ax = fig.gca(projection='3d')

#Plot the surface. 
surf = ax.plot_surface(X_vals,Y_vals,predict,cmap=cm.coolwarm,linewidth=0,antialiased=False)
#surf = ax.plot_surface(X_vals,Y_vals,FrankeFunction(X_vals, Y_vals),cmap=cm.coolwarm,linewidth=0,antialiased=False)

#Customize the z axis. 
ax.set_zlim(-0.10,1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')

#Add a color bar which maps values to colors. 
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


# 1D plot
y_const = 0.5
z = FrankeFunction(x, y_const) + noise

plt.plot(x, z, '.', label='Points')
plt.plot(x_vals, polyval(x_vals, y_const, beta_ols), label='Fitted')
plt.plot(x_vals, FrankeFunction(x_vals, y_const), label='Franke')
plt.legend(loc='best')
plt.show()




# Confidence interval, beta
print(np.var(beta_ols))


# Mean square error (MSE):
MSE = (z - polyval(x, y, beta_ols)).T.dot(z - polyval(x, y, beta_ols))/N
print(MSE)


# R2 score function
denominator = (y-np.mean(y)).T.dot(y-np.mean(y))
print(1-N*MSE/denominator)

'''
N  = 100        # Number of points
D  = 2          # Dimension
Px = 5          # Polynomial order in x-direction
Py = 5          # Polynomial order in y-direction

noise = normal(0,0.0,N)

x = uniform(0,1,N)
y = uniform(0,1,N)
z = FrankeFunction(x, y) #+ noise

beta = reg_q_2D(x, y, z, Px, Py, 1)         # Ridge

x_vals = y_vals = np.linspace(0,1,1000)
X_vals, Y_vals = np.meshgrid(x_vals, y_vals)
predict = f_2D(X_vals, Y_vals, beta)

# PLOT
fig = plt.figure() 
ax = fig.gca(projection='3d')

#Plot the surface. 
surf = ax.plot_surface(X_vals,Y_vals,predict,cmap=cm.coolwarm,linewidth=0,antialiased=False)

#Customize the z axis. 
ax.set_zlim(-0.10,1.40)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

#Add a color bar which maps values to colors. 
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

# 1D plot
y_const = 0.5
z = FrankeFunction(x, y_const) + noise

plt.plot(x, z, '.', label='Points')
plt.plot(x_vals, f_2D(x_vals, y_const, beta), label='Fitted')
plt.plot(x_vals, FrankeFunction(x_vals, y_const), label='Franke')
plt.legend(loc='best')
plt.show()
'''
