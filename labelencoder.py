from sklearn.preprocessing import LabelEncoder


ordinal_data = ['L', 'M', 'M', 'XL', 'XL']  #odinal data


ordinal_categories = ['XL', 'L', 'M']  #order define of the categories


label_encoder = LabelEncoder()   #labelencoder obj. with specified order
label_encoder.fit(ordinal_categories)


encoded_ordinal_data = label_encoder.transform(ordinal_data) #transfrom odinal data using labelencoding


print("Original ordinal data:", ordinal_data)  #display origanl and encoded ordinal data
print("Encoded ordinal data:", encoded_ordinal_data)


decoded_data = label_encoder.inverse_transform(encoded_ordinal_data)  #transform encoded data back to origna categores
print("Decoded data:", decoded_data)
