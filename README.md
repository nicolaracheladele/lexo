# lexo

## Architecture and Flow

1. User selects “seed” readers similar to their taste (e.g., users with similar ratings).

2. Based on those readers, the app recommends 5 books they liked but the current user hasn’t read.

3. The user can rate recommended books and submit ratings.

4. New ratings are appended to the dataset in-memory (or saved for persistence).

5. The recommendation model updates or refines predictions including new ratings (optionally retrain or incrementally update).
