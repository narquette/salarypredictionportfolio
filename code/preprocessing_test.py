from sklearn.preprocessing import OrdinalEncoder
import numpy as np

degree_ord_map = {
    'NONE': 1,
    'HIGH_SCHOOL': 2,
    'BACHELORS': 3,
    'MASTERS': 4,
    'DOCTORAL': 5
}

#degrees = ['NONE', 'HIGH_SCHOOL', 'BACHELORS', 'MASTERS', 'DOCTORAL']
#degrees = np.array(degrees)
# enc = OrdinalEncoder(categories=[np.array(['NONE', 'HIGH_SCHOOL', 'BACHELORS', 'MASTERS', 'DOCTORAL'], dtype=object), 
#                      np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=object)])
enc = OrdinalEncoder(categories=[np.array(list(degree_ord_map.keys()), dtype=object)])

degree_data = [ ['NONE'], ['HIGH_SCHOOL'], ['MASTERS'], 
                ['NONE'], ['MASTERS'], ['BACHELORS'], 
                ['BACHELORS'], ['HIGH_SCHOOL'], ['DOCTORAL'],
                ['DOCTORAL'] ]

print(enc.fit(degree_data))
print(enc.transform(degree_data))
print(list(degree_ord_map.keys()))
