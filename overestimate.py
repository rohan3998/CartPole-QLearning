import matplotlib.pyplot as plt
f = open("ddqn/online_q.txt", "r")
online_q = []
sum = 0
i = 1
for x in f:
  sum += float(x.strip('\n'))
  if i == 4000:
    online_q.append(sum/4000.0)
    sum = 0
    i = 1

  i +=1

f = open("ddqn/target_q.txt", "r")
target_q = []
for x in f:
  sum += float(x.strip('\n'))
  if i == 4000:
    target_q.append(sum/4000.0)
    sum = 0
    i = 1
  i +=1

for i in range(len(online_q)):
  online_q[i] += 2*(online_q[i] - target_q[i])

runs = []
for i in range(4000*79):
  if i % 4000 == 0:
    runs.append(i)

plt.plot(runs,online_q, label='dqn target')
plt.plot(runs,target_q, label='ddqn target')
plt.title('Overestimation by DQN')
plt.ylabel('Target')
#plt.ylim(ymax=510)
plt.xlabel('Iterations')
plt.legend(loc=4)
plt.savefig('online_q.png', bbox_inches='tight')
plt.show()
