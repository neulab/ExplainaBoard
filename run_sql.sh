
# cp xlsum.tsv /opt/lampp/var/mysql/test/

/opt/lampp/bin/mysql -u root --database=test
SET GLOBAL max_allowed_packet=2147483648;
use test;
source /home/ubuntu/ExplainaBoard_Dev/frontend/database/task_ner.sql;
source /home/ubuntu/ExplainaBoard_Dev/frontend/database/task_ner_error.sql;






/opt/lampp/bin/mysql -u root --database=test --execute="LOAD DATA INFILE 'task_ner.tsv' INTO TABLE task_ner FIELDS TERMINATED BY '\t'  ENCLOSED BY '' LINES TERMINATED BY '\n';"
/opt/lampp/bin/mysql -u root --database=test --execute="update task_ner set single = '1'"
/opt/lampp/bin/mysql -u root --database=test --execute="drop table task_ner_error"
/opt/lampp/bin/mysql -u root --database=test --execute="create table task_ner_error like task_ner"
/opt/lampp/bin/mysql -u root --database=test --execute="truncate table task_ner_error"
/opt/lampp/bin/mysql -u root --database=test --execute="LOAD DATA INFILE 'task_ner.tsv' INTO TABLE task_ner_error FIELDS TERMINATED BY '\t'  ENCLOSED BY '' LINES TERMINATED BY '\n';"


