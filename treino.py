from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

def train_model(model, train_gen, val_gen, epochs=100):
    """Treina o modelo com Early Stopping e Checkpoints."""
    early_stopping = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
    checkpoint = ModelCheckpoint('bovino_classification_model.keras', monitor='val_loss', save_best_only=True, mode='min')

    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=[early_stopping, checkpoint])

    return history