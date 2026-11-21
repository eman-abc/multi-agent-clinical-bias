# Case audit

801 cases read.

Legacy `facts` (the exporter's field) against the cleaned facts: {'same': 721, 'truncated': 70, 'stem_retained': 10}. 'truncated' = the legacy text stops early (often inside a number, or before a data block); 'stem_retained' = the legacy text still contains the question stem.

- legacy truncated (70): ['228', '604', '432', '238', '616', '225', '80', '71', '780', '479', '257', '87', '70', '253', '581', '611', '416', '96', '478', '62', '254', '735', '77', '303', '553', '29', '624', '20', '592', '334', '54', '139', '144', '351', '134', '226', '282', '52', '83', '12', '461', '673', '295', '97', '327', '424', '209', '215', '439', '449', '251', '635', '118', '779', '364', '409', '241', '725', '504', '724', '263', '23', '773', '523', '232', '656', '212', '499', '734', '531']
- legacy stem retained (10): ['270', '131', '767', '153', '287', '428', '639', '590', '174', '50']

Legacy cases written from wrong facts (truncated or stem retained) inside the nested samples you already ran: first 20: 3 of 20, first 200: 15 of 200, first 341: 31 of 341, first 801: 80 of 801

EXCLUDED from every arm: 8 (no vignette: ['590', '50']; image-dependent: ['665', '389', '425', '781', '485', '84'])

Ineligible for SEX carriers: 49; RACE: 8; SES: 20.

For manual review (stem not final / not detected): []

Most frequent reasons a case is ineligible:
- sex: [('he', 6), ('gestation', 6), ('her', 5), ('menstrual', 5), ('his', 5), ('she', 4), ('breast', 4), ('gravida', 3), ('newborn', 3), ('para', 3), ('pregnancy', 3), ('contraceptive', 3)]
- race: [('sickle cell', 5), ('hemochromatosis', 2), ('cystic fibrosis', 1)]
- ses: [('homeless', 10), ('fired from', 3), ('shelter', 3), ('immigrated', 2), ('unemployed', 2), ('corrections officers', 1), ('uninsured', 1), ('refugee', 1), ('foreclosure', 1), ('lost their job', 1), ('welfare', 1), ('low-income', 1)]
