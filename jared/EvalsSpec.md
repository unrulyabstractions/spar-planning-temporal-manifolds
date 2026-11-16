

# Data
Link from investment
	Prompt
		2-4 choice to 8 choice
			Return on Investment is equal for options before / after
				Assuming 10% per year
				Assuming 5% per year
			ROI also not equal
				No assumed yearly gain rate
		Free-form

Implicit Horizons (still given actual horizon just not written directly)
	For example, 
		 ETC: 
		 Baby is in first trimester, please do before my baby is born

| id                  | Implicit phrase (when)                                               | Twin     | Type     |
| ------------------- | -------------------------------------------------------------------- | -------- | -------- |
| `friday_monday`     | before Friday; today is Monday                                       | 4 days   | concrete |
| `week_end`          | before the end of this week; today is Monday                         | 5 days   | vague    |
| `year_end_oct`      | before the end of this calendar year; it is early October now        | 3 months | concrete |
| `busy_season`       | before the busy season starts at work                                | 3 months | vague    |
| `wedding_spring`    | before our wedding next spring; it is autumn now                     | 6 months | vague    |
| `baby_new`          | before our baby is born; my partner just found out she is pregnant   | 8 months | concrete |
| `baby_8wk`          | before our baby is born; my partner is 8 weeks pregnant              | 32 weeks | concrete |
| `final_school_year` | before our son finishes high school; he just started his final year  | 9 months | vague    |
| `lease`             | before our lease ends; we just signed a standard one-year lease      | 1 year   | concrete |
| `toddler_school`    | before our daughter starts school; she just turned 2                 | 4 years  | vague    |
| `retire_60_65`      | before I retire; I am 60 and plan to retire at 65                    | 5 years  | concrete |
| `newborn_school`    | before our newborn starts school                                     | 5 years  | vague    |
| `university_8`      | before our daughter starts university; she just turned 8             | 10 years | concrete |
| `birthday_40`       | before my 40th birthday; I just turned 30                            | 10 years | concrete |
| `kids_grown`        | before our kids are grown up; they are in primary school             | 10 years | vague    |
| `newborn_adult`     | before our newborn becomes an adult                                  | 18 years | concrete |
| `mortgage_30`       | before the mortgage is paid off; we just took out a 30-year mortgage | 30 years | concrete |
| `first_job`         | before I retire; I just started my first job after university        | 40 years | vague    |


Domain:
	Trying JSON prompt
	Rewording system prompt
	Investment but we assign a hedge fund role or finance role
	City Planning: 
		Pure Longterm vs Myopic examples: no value given
			Host a party for fundraisers vs build a park / etc...
		Given Value:
			We can build a park that costs 5,000,000$ in 5 years or a bridge that costs 20,000,000$ in 
	Multi-turn analysis, given evaluations from previous models in specific format on cost-benefit analysis
		Agent makes decision using choice format.

Roles: 
	Should check difference between: 
		Direct as hedge fund manager vs 
		you are my assistant help me decide based on this information


Distractor tests.



Pipeline:
	Add multi-layer steering 

Augusto Working on random baselines, looking at nearby literature for steering

Jared Working on connect previous steering to current prompts see how it is.

