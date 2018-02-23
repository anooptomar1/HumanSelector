import turicreate as tc

data = tc.image_analysis.load_images('train', with_path=True)
data['label'] = data['path'].apply(lambda path: 'human' if 'human' in path else 'not a human')
data.save('peopleornot4.sframe')
train_data, test_data = data.random_split(0.8)
model = tc.image_classifier.create(train_data, target='label')
predictions = model.predict(test_data)
metrics = model.evaluate(test_data)
print(metrics['accuracy'])
model.save('peopleornot4.model')
model.export_coreml('peopleornot4.mlmodel')
