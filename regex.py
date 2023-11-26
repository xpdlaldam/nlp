import re

#### 09/17/2023
###
text1 = 'codebasics: you ask a lot of questions 1234551585, a23bc@yahoo.com'
re.findall('\d{5}', text1)

### Because the parenthesis regex captures everything enclosed, we need to escape with \ to get the literal (123 )
text2 = '(123)-123-1234, abX_82@gmail.com abc@gmaildotcom'
re.findall('\(\d{3}\)', text2)
re.findall('\(\d{3}\)-\d{3}-\d{4}', text2)

###
re.findall('\d{10}|\(\d{3}\)-\d{3}-\d{4}', text1)
re.findall('\d{10}|\(\d{3}\)-\d{3}-\d{4}', text2)

### 
## * means a sequence CRUCIAL
## . means a single character => hence we need to escape for literal . CRUCIAL
re.findall('[a-z0-9]*@[a-z]*.com', text1)
re.findall('[a-zA-Z0-9_]*@[a-z]*\.com', text2)

### CRUCIAL: How to capture a portion of an entire regex pattern => ans: by using ()
chat1 = 'my order # 412889912'
chat2 = 'my order number 412889912'
chat3 = 'my order is 412889912'

# [^\d]*: anything except for a sequence of digits (this way we can capture spaces too)
re.findall('order[^\d]*', chat1)
re.findall('order[^\d]*', chat2)
re.findall('order[^\d]*', chat3)

# Now it captures the digits we want but we only want the digits
re.findall('order[^\d]*\d*', chat1)

# We surround () to the portion of the pattern of interest
re.findall('order[^\d]*(\d*)', chat1)
re.findall('order[^\d]*(\d*)', chat2)
re.findall('order[^\d]*(\d*)', chat3)

###
t1 = 'Born    Elon Reeve Musk'
re.findall('Born(.*)', t1)
re.findall('Born(.*)', t1)[0].strip()

### 
t2 = '''
Born    Elon Reeve Musk
June 28, 1971 (age 50)
Madison, WI, USA
'''

# Grab 50
re.findall('\(age.*(\d{2})', t2) # or
re.findall('age (\d+)', t2)

# Grab the entire line after "Born    Elon Reeve Musk"
re.findall('Born.*\n(.*)', t2)

# Grab "June 28, 1971"
re.findall('Born.*\n(.*)\(age', t2)[0].strip()

# Grab "Madison, WI, USA"
re.findall('\(age.*\n(.*)', t2)

# Grab "Madison"
re.findall('\(age.*\n(.*), \D+, \D+', t2)

# Grab "WI"
re.findall('\(age.*\n.*, (\D+), \D+', t2)

# Grab "USA"
re.findall('\(age.*\n.*, \D+, (\D+)', t2)[0].strip()