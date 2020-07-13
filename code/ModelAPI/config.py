import pandas as pd
import numpy as np
import os

# load test data
test_data = pd.read_csv(os.path.join('../','../data','original','test_features.tar.gz'), compression='gzip')
test_data.dropna(inplace=True)

# create content for web form for companyId
company_ids_unique = set(test_data['companyId'])
company_ids = [(i, comp) for i, comp in enumerate(sorted(company_ids_unique))]

# create content for web form for job type
job_types_unique = set(test_data['jobType'])
job_types = [(i, job) for i, job in enumerate(sorted(job_types_unique))]

# create content for web form for degree
degree_unique = set(test_data['degree'])
degrees = [(i, degree) for i, degree in enumerate(sorted(degree_unique))]

# create content for web form for major
major_unique = set(test_data['major'])
majors = [(i, major) for i, major in enumerate(sorted(major_unique))]

# create content for web form for major
industry_unique = set(test_data['industry'])
industries = [(i, industry) for i, industry in enumerate(sorted(industry_unique))]
