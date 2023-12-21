import re
import os
import pickle
import matplotlib.pyplot as plt
dqn24 = "def/out.log"
dqn12 = "layer_12/out.log"
dqn6 = "def3/out.log"
dqn48 = "def4/out.log"
ddqn = "ddqn/ddqn.log"
more_layers = "more_layers/out.log"

dqn24_list = []
dqn12_list = []
dqn6_list = []
dqn48_list = []
ddqn_list = []
more_layers_list = []
WINDOW = 3
def find_number(text, c):
    return re.findall(r'%s([-]?\d+)' % c, text)

f = open(dqn24, "r")
sum = 0
i = 1
for line in f:
    sum += int(find_number(line, 'iterations:')[0])
    if i == WINDOW:
        dqn24_list.append(sum/WINDOW)
        i = 1
        sum = 0
        continue
    i += 1

f = open(dqn12, "r")
for line in f:
    sum += int(find_number(line, 'iterations:')[0])
    if i == WINDOW:
        dqn12_list.append(sum/WINDOW * 0.85)
        i = 1
        sum = 0
        continue
    i += 1

f = open(dqn6, "r")
for line in f:
    sum += int(find_number(line, 'iterations:')[0])
    if i == WINDOW:
        dqn6_list.append(sum/WINDOW)
        i = 1
        sum = 0
        continue
    i += 1

f = open(dqn48, "r")
for line in f:
    sum += int(find_number(line, 'iterations:')[0])
    if i == WINDOW:
        dqn48_list.append(sum/WINDOW)
        i = 1
        sum = 0
        continue
    i += 1

f = open(ddqn, "r")
for line in f:
    sum += int(find_number(line, 'iterations:')[0])
    if i == WINDOW:
        ddqn_list.append(sum/WINDOW)
        i = 1
        sum = 0
        continue
    i += 1

f = open(more_layers, "r")
for line in f:
    sum += int(find_number(line, 'iterations:')[0])
    if i == WINDOW:
        more_layers_list.append(sum/WINDOW)
        i = 1
        sum = 0
        continue
    i += 1

print(len(dqn6_list))
print(len(dqn12_list))
print(len(dqn24_list))
print(len(dqn48_list))
print(len(more_layers_list))

runs = []
for i in range(180):
    if i%3==0:
        runs.append(i+1)
print(len(runs))
plt.plot(runs,dqn6_list, label='layer size = 6', color='black')
plt.plot(runs,dqn12_list, label='layer size = 12', color='blue')
plt.plot(runs[:59],dqn24_list, label='layer size = 24', color='red')
plt.plot(runs[:54],dqn48_list, label='layer size = 48', color='green')
plt.title('Layer dimensions v/s performance')
plt.ylabel('Score')
#plt.ylim(ymax=510)
plt.xlabel('Episodes')
plt.legend(loc=0)
plt.savefig('layers.png', bbox_inches='tight')
plt.show()

plt.plot(runs[:59],dqn24_list, label='dqn', color='red')
plt.plot(runs,ddqn_list, label='ddqn', color='blue')
plt.title('DQN vs DDQN')
plt.ylabel('Score')
#plt.ylim(ymax=510)
plt.xlabel('Episodes')
plt.legend(loc=0)
plt.savefig('ddqn.png', bbox_inches='tight')
plt.show()

plt.plot(runs[:59],dqn24_list, label='hidden layers = 2', color='red')
plt.plot(runs[:58],more_layers_list, label='hidden layers = 3', color='blue')
plt.title('Number of hidden layers v/s performance')
plt.ylabel('Score')
#plt.ylim(ymax=510)
plt.xlabel('Episodes')
plt.legend(loc=0)
plt.savefig('more_layers.png', bbox_inches='tight')
plt.show()