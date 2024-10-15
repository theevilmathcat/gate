#!/bin/bash

# Define backup directory
BACKUP_DIR="/backup"  # This will be a mounted volume
TIMESTAMP=$(date +"%F_%H-%M-%S")
BACKUP_FILE="$BACKUP_DIR/db_backup_$TIMESTAMP.sql"

# Create the backup using pg_dump
pg_dump -U ${POSTGRES_USER} -h db -d ${POSTGRES_DB} > "$BACKUP_FILE"

# Optional: Compress the backup file
gzip "$BACKUP_FILE"

echo "Backup completed: $BACKUP_FILE.gz"
