# 1. DO NOT use %matplotlib inline. Use the terminal graphics backend instead:
import matplotlib
matplotlib.use('module://kitcat') # Or 'module://matplotlib_backend_kitty'

# 2. Import your standard tools
import seaborn as sns
import matplotlib.pyplot as plt

# 3. Build your plot
tips = sns.load_dataset("tips")
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="smoker")

# 4. Trigger the render sequence 
plt.show()
