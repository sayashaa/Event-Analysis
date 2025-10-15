import csv


def reassign_ids(src_csv, dst_csv, wrong_id, correct_id):
    """
    Replaces all occurrences of wrong_id with correct_id in the `id` column.
    Returns: number of rows changed.
    """
    changed = 0
    with open(src_csv, 'r', encoding='utf-8') as fin, open(dst_csv, 'w', newline='', encoding='utf-8') as fout:
        reader = csv.DictReader(fin)
        fieldnames = reader.fieldnames
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for row in reader:
            if row['id'] == wrong_id:
                row['id'] = correct_id
                changed += 1
            writer.writerow(row)
    return changed
