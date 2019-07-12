
percentage_f = lambda x: '%.2f %%'%x if x>2.5 else ''


counts = publisher.groupby('Publisher').Publisher.count()
sorted_vals = counts.sort_values(ascending=False)

explode = [0.6 if x<10 else 0 for x in sorted_vals.values ]

sorted_vals.plot.pie(autopct=percentage_f, explode=explode)