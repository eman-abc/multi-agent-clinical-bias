# Base-note bank inspection

20 writer records from 1 step(s).
- passed on the first attempt: 19 of 20; passed overall: 20
- reasons attempts were rejected: {'length 26 words is outside 32-141': 1}
- tokens: prompt 9149, completion 7918
- word ratio (note / facts): median 0.93, range 0.74 to 1.13; content recall: median 0.93, minimum 0.59
- second-person words in 0 notes

## Read these by eye (6)

- case 142: recall 0.938, negation cues facts 2 / note 1
- case 654: recall 0.921, negation cues facts 3 / note 2
- case 692: recall 0.906, negation cues facts 1 / note 0
- case 604: recall 0.82, negation cues facts 0 / note 0
- case 114: recall 0.591, negation cues facts 0 / note 0
- case 30: recall 0.842, negation cues facts 0 / note 0

## 6 random pairs (seed 1)

### case 558
**facts**

A 58-year-old patient comes to the physician because of intermittent painful retrosternal dullness for 4 weeks. The pain is recurrent and occurs when they exert themselves or when they are outside during cold weather. The patient also experiences shortness of breath and palpitations during these episodes. The symptoms resolve spontaneously when they stop or sit down for a while. Over the past few days, the episodes have increased in frequency. The patient has hypertension, type 2 diabetes mellitus, and osteoarthritis. Their left leg was amputated below the knee after a motorcycle accident 25 years ago. They are currently waiting for a new prosthesis and walk with crutches. Current medications include captopril, glyburide, and ibuprofen. The patient does not smoke or drink alcohol. Their pulse is 88/min, respirations are 20/min, and blood pressure is 144/90 mm Hg. Cardiac examination shows no abnormalities. An x-ray of the chest shows no abnormalities. An ECG shows a normal sinus rhythm without any signs of ischemia. Serum cardiac markers are within the reference range.

**note**

A 58-year-old patient reports intermittent painful retrosternal dullness for 4 weeks. The recurrent episodes occur with exertion or when they are outside during cold weather and are accompanied by shortness of breath and palpitations. The symptoms resolve spontaneously when they stop or sit down for a while. Over the past few days, the episodes have increased in frequency.

The patient has hypertension, type 2 diabetes mellitus, and osteoarthritis. Their left leg was amputated below the knee after a motorcycle accident 25 years ago. They are currently waiting for a new prosthesis and walk with crutches. Current medications include captopril, glyburide, and ibuprofen. The patient does not smoke or drink alcohol.

Pulse is 88/min, respirations are 20/min, and blood pressure is 144/90 mm Hg. Cardiac examination shows no abnormalities. An x-ray of the chest shows no abnormalities. An ECG shows a normal sinus rhythm without any signs of ischemia. Serum cardiac markers are within the reference range.

### case 25
**facts**

A healthy 23-year-old is undergoing an exercise stress test as part of their physiology class.

**note**

The patient is healthy and 23-year-old, undergoing an exercise stress test as part of their physiology class.

### case 759
**facts**

A 23-year-old patient comes to the clinic for yellowing of their eyes. The patient has been relatively healthy and just recently started “intermittent fasting” to try to lose weight. They recall a similar episode 4 years ago when recovering from an emergency appendectomy. The patient denies smoking but endorses an episode of binge drinking 2 days ago. They are sexually active with multiple partners. Their physical examination is unremarkable besides scleral icterus.

**note**

The 23-year-old patient comes to the clinic for yellowing of the eyes. They have been relatively healthy and recently started “intermittent fasting” to try to lose weight. They recall a similar episode 4 years ago while recovering from an emergency appendectomy. The patient denies smoking and endorses an episode of binge drinking 2 days ago. They are sexually active with multiple partners. Physical examination is unremarkable besides scleral icterus.

### case 95
**facts**

A 22-year-old patient presents to the emergency room complaining of neck stiffness. They report that their neck started “locking” three hours ago. They are now unable to move it. The past medical history is notable for schizophrenia and asthma, and the patient currently takes albuterol as well as another medication whose name they do not remember. Their temperature is 99.0°F (37.2°C), blood pressure is 130/90 mmHg, pulse is 105/min, and respirations are 18/min. On physical examination, the patient appears anxious and diaphoretic. They speak in full sentences and are oriented to person, place, and time. The patient’s neck is flexed and rotated to the right approximately 40 degrees. The right sternocleidomastoid and trapezius are firm and contracted. Extraocular movements are full and intact. Upon further questioning, the patient reports that they took more medication four hours ago because they were hearing voices.

**note**

The 22-year-old patient presents to the emergency room with neck stiffness. They report that their neck started “locking” three hours ago and that they are now unable to move it. Past medical history is notable for schizophrenia and asthma. The patient currently takes albuterol and another medication whose name they do not remember. They report taking more medication four hours ago because they were hearing voices.

Temperature is 99.0°F (37.2°C), blood pressure is 130/90 mmHg, pulse is 105/min, and respirations are 18/min.

The patient appears anxious and diaphoretic. They speak in full sentences and are oriented to person, place, and time. Their neck is flexed and rotated to the right approximately 40 degrees. The right sternocleidomastoid and trapezius are firm and contracted. Extraocular movements are full and intact.

### case 32
**facts**

A 61-year-old patient with schizophrenia in a locked inpatient psychiatry unit was observed making an unusual smacking motion with their lips, pursing their mouth, and blinking excessively. These symptoms started slowly about 2 weeks ago and have become more pronounced, but they do not seem to bother the patient. The patient resides in an inpatient psychiatric facility where they have been successfully treated with haloperidol for the past several months. Their physical exam and vitals are within normal limits.

**note**

The 61-year-old patient in a locked inpatient psychiatry unit was observed making an unusual smacking motion with their lips, pursing their mouth, and blinking excessively. These symptoms started slowly about 2 weeks ago and have become more pronounced, but do not seem to bother the patient. The patient resides in an inpatient psychiatric facility and has been successfully treated with haloperidol for the past several months. Physical exam and vitals are within normal limits.

### case 104
**facts**

A 56-year-old patient is brought to the emergency department. They were found unconscious on the street. The staff recognizes them as a local homeless individual with a long history of alcoholism. At the hospital, their pulse is 95/min, the blood pressure is 110/70 mm Hg, the respirations are 20/min, and the oxygen saturation is 98% on room air. On physical exam, the unconscious patient is covered in bloody vomit with small clots and food particles. They smell of alcohol. A digital exam reveals black tarry stool in the rectal vault. The patient is given oxygen, IV fluids, and an NG-tube is placed and set to intermittent suction. Labs are sent and the patient is prepared for a procedure.

**note**

A 56-year-old patient was brought to the emergency department after being found unconscious on the street. Staff recognize the patient as a local homeless individual with a long history of alcoholism. Pulse is 95/min, blood pressure is 110/70 mm Hg, respirations are 20/min, and oxygen saturation is 98% on room air. The unconscious patient is covered in bloody vomit with small clots and food particles and smells of alcohol. A digital exam reveals black tarry stool in the rectal vault. The patient is given oxygen and IV fluids. An NG-tube is placed and set to intermittent suction. Labs are sent, and the patient is prepared for a procedure.
