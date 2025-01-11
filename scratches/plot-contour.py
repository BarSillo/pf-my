# Contour plot of Realized Delta in Volatility Units
plt.figure(figsize=(10, 6))
X, Y = np.meshgrid(log_moneyness_range, volatility_range)
contour = plt.contourf(X, Y, delta_changes_volatility_units, levels=50, cmap='viridis')
plt.colorbar(contour, label='Realized Delta (in Volatility Units)')
plt.xlabel('Log-Moneyness (log(S/K))')
plt.ylabel('Volatility')
plt.title('Contour Plot of Realized Delta in Volatility Units')
plt.show()
