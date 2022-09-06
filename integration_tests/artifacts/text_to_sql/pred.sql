select count(*) from singer	concert_singer
select count(*) from singer	concert_singer
select name, country, age from singer order by age desc	concert_singer
select name, country, age from singer order by age desc	concert_singer
select avg(age), min(age), max(age) from singer where country = "France"	concert_singer
select avg(age), min(age), max(age) from singer where country = "France"	concert_singer
select song_name, song_release_year from singer order by age limit 1	concert_singer
select song_name, song_release_year from singer order by age limit 1	concert_singer
select distinct country from singer where age > 20	concert_singer
select distinct country from singer where age > 20	concert_singer
select country, count(*) from singer group by country	concert_singer
select country, count(*) from singer group by country	concert_singer
select song_name from singer where age > (select avg(age) from singer)	concert_singer
select song_name from singer where age > (select avg(age) from singer)	concert_singer
select location, name from stadium where capacity between 5000 and 10000	concert_singer
select location, name from stadium where capacity between 5000 and 10000	concert_singer
select max(capacity), avg(capacity) from stadium	concert_singer
select avg(capacity), max(capacity) from stadium	concert_singer
select name, capacity from stadium order by average desc limit 1	concert_singer
select name, capacity from stadium order by average desc limit 1	concert_singer
select count(*) from concert where year = 2014 or year = 2015	concert_singer
select count(*) from concert where year = 2014 or year = 2015	concert_singer
select t1.name, count(*) from stadium as t1 join concert as t2 on t1.stadium_id = t2.stadium_id group by t1.stadium_id	concert_singer
select t1.name, count(*) from stadium as t1 join concert as t2 on t1.stadium_id = t2.stadium_id group by t1.stadium_id	concert_singer
select t1.name, t1.capacity from stadium as t1 join concert as t2 on t1.stadium_id = t2.stadium_id where t2.year >= 2014 group by t2.stadium_id order by count(*) desc limit 1	concert_singer
select t1.name, t1.capacity from stadium as t1 join concert as t2 on t1.stadium_id = t2.stadium_id where t2.year > 2013 group by t2.stadium_id order by count(*) desc limit 1	concert_singer
select year from concert group by year order by count(*) desc limit 1	concert_singer
select year from concert group by year order by count(*) desc limit 1	concert_singer
select name from stadium where stadium_id not in (select stadium_id from concert)	concert_singer
select name from stadium where stadium_id not in (select stadium_id from concert)	concert_singer
select country from singer where age > 40 intersect select country from singer where age < 30	concert_singer
select name from stadium except select t2.name from concert as t1 join stadium as t2 on t1.stadium_id = t2.stadium_id where t1.year = 2014	concert_singer
select name from stadium except select t2.name from concert as t1 join stadium as t2 on t1.stadium_id = t2.stadium_id where t1.year = 2014	concert_singer
select t1.concert_name, t1.theme, count(*) from concert as t1 join singer_in_concert as t2 on t1.concert_id = t2.singer_id group by t1.concert_id	concert_singer
select t1.concert_name, t1.theme, count(*) from concert as t1 join singer_in_concert as t2 on t1.concert_id = t2.singer_id group by t1.concert_id	concert_singer
select t2.name, count(*) from singer_in_concert as t1 join singer as t2 on t1.singer_id = t2.singer_id group by t1.singer_id	concert_singer
select t2.name, count(*) from singer_in_concert as t1 join singer as t2 on t1.singer_id = t2.singer_id group by t1.singer_id	concert_singer
select t2.name from singer_in_concert as t1 join singer as t2 on t1.singer_id = t2.singer_id join concert as t3 on t3.concert_id = t1.concert_id where t3.year = 2014	concert_singer
select t2.name from singer_in_concert as t1 join singer as t2 on t1.singer_id = t2.singer_id join concert as t3 on t3.concert_id = t1.concert_id where t3.year = 2014	concert_singer
select name, country from singer where song_name like '%hey%'	concert_singer
select name, country from singer where song_name like '%hey%'	concert_singer
select t1.name, t1.location from stadium as t1 join concert as t2 on t1.stadium_id = t2.stadium_id where t2.year = 2014 intersect select t1.name, t1.location from stadium as t1 join concert as t2 on t1.stadium_id = t2.stadium_id where t2.year = 2015	concert_singer
select t1.name, t1.location from stadium as t1 join concert as t2 on t1.stadium_id = t2.stadium_id where t2.year = 2014 intersect select t1.name, t1.location from stadium as t1 join concert as t2 on t1.stadium_id = t2.stadium_id where t2.year = 2015	concert_singer
select count(*) from concert as t1 join stadium as t2 on t1.stadium_id = t2.stadium_id order by t2.capacity desc limit 1	concert_singer
select count(*) from concert as t1 join stadium as t2 on t1.stadium_id = t2.stadium_id order by t2.capacity desc limit 1	concert_singer
select count(*) from pets where weight > 10	pets_1
select count(*) from pets where weight > 10	pets_1
select weight from pets order by pet_age asc limit 1	pets_1
select weight from pets where pettype = 'dog' and pet_age = (select min(pet_age) from pets)	pets_1
select max(weight), pettype from pets group by pettype	pets_1
select max(weight), pettype from pets group by pettype	pets_1
select count(*) from has_pet as t1 join pets as t2 on t1.petid = t2.petid join student as t3 on t3.stuid = t1.stuid where t3.age > 20	pets_1
select count(*) from has_pet as t1 join pets as t2 on t1.petid = t2.petid join student as t3 on t1.stuid = t3.stuid where t3.age > 20	pets_1
select count(*) from pets as t1 join has_pet as t2 on t1.petid = t2.petid join student as t3 on t2.stuid = t3.stuid where t3.sex = "F" and t1.pettype = "dog"	pets_1
select count(*) from pets as t1 join has_pet as t2 on t1.petid = t2.petid join student as t3 on t2.stuid = t3.stuid where t3.sex = "F" and t1.pettype = "dog"	pets_1
select count(distinct pettype) from pets	pets_1
select count(distinct pettype) from pets	pets_1
select t1.fname from student as t1 join has_pet as t2 on t1.stuid = t2.stuid join pets as t3 on t2.petid = t3.petid where t3.pettype = "cat" or t3.pettype = "dog"	pets_1
select t1.fname from student as t1 join has_pet as t2 on t1.stuid = t2.stuid join pets as t3 on t2.petid = t3.petid where t3.pettype = "cat" or t3.pettype = "dog"	pets_1
select fname from student where stuid in (select petid from pets where pettype = "cat") intersect select fname from student where stuid in (select petid from pets where pettype = "dog")	pets_1
select fname from student where stuid in (select petid from pets where pettype = "cat") intersect select fname from student where stuid in (select petid from pets where pettype = "dog")	pets_1
select major, age from student except select t1.major, t1.age from student as t1 join has_pet as t2 on t1.stuid = t2.stuid join pets as t3 on t2.petid = t3.petid where t3.pettype = "cat"	pets_1
select major, age from student where stuid not in (select petid from pets where pettype = "cat")	pets_1
select stuid from student except select t1.stuid from has_pet as t1 join pets as t2 on t1.petid = t2.petid where t2.pettype = "cat"	pets_1
select stuid from student except select stuid from has_pet join pets on pets.petid = pets.petid where pets.pettype = "cat"	pets_1
select fname, age from student where stuid in (select petid from pets where pettype = "dog") except select fname, age from student where stuid in (select petid from pets where pettype = "cat")	pets_1
select t1.fname from student as t1 join has_pet as t2 on t1.stuid = t2.stuid join pets as t3 on t2.petid = t3.petid where t3.pettype = "dog" except select t1.fname from student as t1 join has_pet as t2 on t1.stuid = t2.stuid join pets as t3 on t2.petid = t3.petid where t3.pettype = "cat"	pets_1
select pettype, weight from pets order by pet_age asc limit 1	pets_1
select pettype, weight from pets order by pet_age asc limit 1	pets_1
select petid, weight from pets where pet_age > 1	pets_1
select petid, weight from pets where pet_age > 1	pets_1
select pettype, avg(pet_age), max(pet_age) from pets group by pettype	pets_1
select pet_age, avg(pet_age), max(pet_age), pettype from pets group by pettype	pets_1
select pettype, avg(weight) from pets group by pettype	pets_1
select pettype, avg(weight) from pets group by pettype	pets_1
select t1.fname, t1.age from student as t1 join has_pet as t2 on t1.stuid = t2.stuid	pets_1
select distinct t1.fname, t1.age from student as t1 join has_pet as t2 on t1.stuid = t2.stuid	pets_1
select t2.petid from student as t1 join has_pet as t2 on t1.stuid = t2.stuid where t1.lname = "Smith"	pets_1
select t2.petid from student as t1 join has_pet as t2 on t1.stuid = t2.stuid where t1.lname = 'Smith'	pets_1
select count(*), stuid from has_pet group by stuid having count(*) >= 1	pets_1
select count(*), t1.stuid from student as t1 join has_pet as t2 on t1.stuid = t2.stuid group by t1.stuid	pets_1
select t1.fname, t1.sex from student as t1 join has_pet as t2 on t1.stuid = t2.stuid group by t1.stuid having count(*) > 1	pets_1
select t1.fname, t1.sex from student as t1 join has_pet as t2 on t1.stuid = t2.stuid group by t1.stuid having count(*) > 1	pets_1
select lname from student where stuid in (select t1.stuid from has_pet as t1 join pets as t2 on t1.petid = t2.petid where t2.pet_age = 3 and t2.pettype = "cat")	pets_1
select t1.lname from student as t1 join has_pet as t2 on t1.stuid = t2.stuid join pets as t3 on t2.petid = t3.petid where t3.pet_age = 3	pets_1
select avg(age) from student where stuid not in (select stuid from has_pet)	pets_1
select avg(age) from student where stuid not in (select stuid from has_pet)	pets_1
select count(*) from continents	car_1
select count(*) from continents	car_1
select t1.contid, t1.continent, count(*) from continents as t1 join countries as t2 on t1.continent = t2.continent group by t1.continent	car_1
select t1.continent, t2.countryname, count(*) from continents as t1 join countries as t2 on t1.continent = t2.continent group by t1.continent	car_1
select count(*) from countries	car_1
select count(*) from countries	car_1
select t1.fullname, t1.id, count(*) from car_makers as t1 join model_list as t2 on t1.id = t2.maker group by t1.id	car_1
select t1.fullname, t1.id, count(*) from car_makers as t1 join model_list as t2 on t1.id = t2.maker group by t1.id	car_1
select t1.model from car_names as t1 join cars_data as t2 on t1.make = t2.id order by t2.horsepower limit 1	car_1
select t1.model from car_names as t1 join cars_data as t2 on t1.make = t2.id order by t2.horsepower limit 1	car_1
select t1.model from car_names as t1 join model_list as t2 on t1.makeid = t2.modelid join cars_data as t3 on t3.id = t2.	car_1
select t2.model from cars_data as t1 join model_list as t2 on t1.id = t2.modelid where t1.weight < (select avg(weight) from cars_data)	car_1
select t2.maker from cars_data as t1 join car_makers as t2 on t1.id = t2.maker where t1.year = 1970	car_1
select distinct t1.maker from car_makers as t1 join cars_data as t2 on t1.maker = t2.id where t2.year = 1970	car_1
select t1.make, t2.year from car_names as t1 join cars_data as t2 on t1.make = t2.id order by t2.year limit 1	car_1
select t1.maker, t2.year from car_makers as t1 join cars_data as t2 on t1.id = t2.id order by t2.year limit 1	car_1
select distinct t1.model from car_names as t1 join cars_data as t2 on t1.makeid = t2.id where t2.year > 1980	car_1
select distinct t2.model from cars_data as t1 join model_list as t2 on t1.id = t2.modelid where t1.year > 1980	car_1
select count(*), continent from continents group by continent	car_1
select t1.continent, count(*) from continents as t1 join car_makers as t2 on t1.contid = t2.maker group by t1.continent	car_1
select t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country group by t1.countryname order by count(*) desc limit 1	car_1
select t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country group by t1.countryname order by count(*) desc limit 1	car_1
select count(*), t1.fullname from car_makers as t1 join model_list as t2 on t1.id = t2.maker group by t1.fullname	car_1
select count(*), t1.id, t1.fullname from car_makers as t1 join model_list as t2 on t1.id = t2.maker group by t1.id	car_1
select t2.accelerate from car_names as t1 join cars_data as t2 on t1.make = t2.id where t1.make = "amc hornet sportabout (sw)"	car_1
select t1.accelerate from cars_data as t1 join car_names as t2 on t1.id = t2.makeid where t2.make = 'amc hornet sportabout (sw)'	car_1
select count(*) from countries as t1 join car_makers as t2 on t1.countryid = t2.country where t1.countryname = "france"	car_1
select count(*) from countries as t1 join car_makers as t2 on t1.countryid = t2.country where t1.countryname = "France"	car_1
select count(*) from countries as t1 join car_makers as t2 on t1.countryid = t2.country where t1.countryname = "usa"	car_1
select count(*) from car_makers where country = 'USA'	car_1
select avg(mpg) from cars_data where cylinders = 4	car_1
select avg(mpg) from cars_data where cylinders = 4	car_1
select min(weight) from cars_data where year = 1974 and cylinders = 8	car_1
select min(weight) from cars_data where cylinders = 8 and year = 1974	car_1
select maker, model from model_list	car_1
select maker, model from model_list	car_1
select t1.countryname, t2.id from countries as t1 join car_makers as t2 on t1.countryid = t2.id	car_1
select t1.countryname, t1.countryid from countries as t1 join car_makers as t2 on t1.countryid = t2.country	car_1
select count(*) from cars_data where horsepower > 150	car_1
select count(*) from cars_data where horsepower > 150	car_1
select avg(weight), year from cars_data group by year	car_1
select avg(weight), avg(year) from cars_data group by year	car_1
select t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country where t1.continent = "europe" group by t1.countryname having count(*) >= 3	car_1
select t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country where t1.continent = "europe" group by t1.countryname having count(*) >= 3	car_1
select max(t2.horsepower), t1.make from car_names as t1 join cars_data as t2 on t1.model = t2.id where t2.cylinders = 3	car_1
select max(horsepower), t1.make from car_names as t1 join model_list as t2 on t1.model = t2.modeli	car_1
select max(mpg) from cars_data	car_1
select max(mpg) from cars_data	car_1
select avg(horsepower) from cars_data where year < 1980	car_1
select avg(horsepower) from cars_data where year < 1980	car_1
select avg(edispl) from cars_data as t1 join model_list as t2 on t1.id = t2.modelid where t2.model = "volvo"	car_1
select avg(edispl) from cars_data as t1 join model_list as t2 on t1.id = t2.modelid where t2.maker = "Volvo"	car_1
select max(accelerate), cylinders from cars_data group by cylinders	car_1
select max(accelerate), cylinders from cars_data group by cylinders	car_1
select model, make from car_names group by model, make order by count(*) desc limit 1	car_1
select model from model_list group by model order by count(*) desc limit 1	car_1
select count(*) from cars_data where cylinders > 4	car_1
select count(*) from cars_data where cylinders > 4	car_1
select count(*) from cars_data where year = 1980	car_1
select count(*) from cars_data where year = 1980	car_1
select count(*) from car_makers where fullname = "American Motor Company"	car_1
select count(*) from car_makers as t1 join model_list as t2 on t1.id = t2.maker where t1.fullname = "American Motor Company"	car_1
select t1.fullname, t1.id from car_makers as t1 join model_list as t2 on t1.id = t2.maker group by t1.id having count(*) > 3	car_1
select t1.fullname, t1.id from car_makers as t1 join model_list as t2 on t1.id = t2.maker group by t1.id having count(*) > 3	car_1
select distinct t1.model from model_list as t1 join car_makers as t2 on t1.maker = t2.id where t2.fullname = "General Motors" or t1.weight	car_1
select distinct t1.model from model_list as t1 join car_makers as t2 on t1.maker = t2.id where t2.fullname = "General Motors" union select distinct t1.model from model_list as t1 join cars_data as t2 on t1.model = t2.id where t2.weight > 3500	car_1
select year from cars_data where weight between 3000 and 4000	car_1
select distinct year from cars_data where weight < 4000 intersect select distinct year from cars_data where weight > 3000	car_1
select horsepower from cars_data order by accelerate desc limit 1	car_1
select horsepower from cars_data order by accelerate desc limit 1	car_1
select cylinders from cars_data where model = 'volvo' and accelerate = (select min(accelerate) from cars_data where model = 'volvo'	car_1
select cylinders from cars_data as t1 join model_list as t2 on t1.id = t2.modelid where t2.maker = "Volvo" order by t1.accelerate limit 1	car_1
select count(*) from cars_data where accelerate > (select max(accelerate) from cars_data order by horsepower desc limit 1)	car_1
select count(*) from cars_data where accelerate > (select max(accelerate) from cars_data order by horsepower desc limit 1)	car_1
select count(*) from (select country from car_makers group by country having count(*) > 2)	car_1
select count(*) from (select country from car_makers group by country having count(*) > 2)	car_1
select count(*) from cars_data where cylinders > 6	car_1
select count(*) from cars_data where cylinders > 6	car_1
select max(horsepower) from cars_data where cylinders = 4	car_1
select t1.model from car_names as t1 join cars_data as t2 on t1.model = t2.id where t2.cylinders = 4 order by t2.horsepower desc limit 1	car_1
select t1.makeid, t1.make from car_names as t1 join cars_data as t2 on t1.model = t2.id where t2.horsepower > (select min(horsepower) from cars_data) except select t1.makeid, t1.make from car_names as t1 join cars_data as t2 on t1.model = t2.id where t2.cylinders > 3	car_1
select t1.makeid, t1.make from car_names as t1 join cars_data as t2 on t1.makeid = t2.id where t2.cylinders < 4	car_1
select max(mpg) from cars_data where cylinders = 8 or year < 1980	car_1
select max(mpg) from cars_data where cylinders = 8 or year < 1980	car_1
select t1.model from model_list as t1 join car_makers as t2 on t1.maker = t2.id where t1.model < 3500 except select t1.model from model_list as t1 join car_makers as t2 on t1.maker = t2.id where t2.fullname = 'Ford Motor Company'	car_1
select distinct t1.model from model_list as t1 join cars_data as t2 on t1.modelid = t2.id where t2.weight < 3500 except select distinct t1.model from model_list as t1 join car_makers as t2 on t1.maker = t2.id where t2.fullname = "Ford Motor Company"	car_1
select countryname from countries where countryid not in (select country from car_makers)	car_1
select countryname from countries except select t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country	car_1
select id, maker from car_makers group by id having count(*) >= 2	car_1
select t1.id, t1.maker from car_makers as t1 join model_list as t2 on t1.id = t2.maker group by t1.id having count(*) >= 2	car_1
select t1.countryid, t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country where t2.maker = "Fiat" group by t1.countryid having count(*) > 3 union select t1.countryid, t1.countryname from countries as t1 join model_list as t2 on t1.countryid = t2.modelid where t2.model = "fiat"	car_1
select t1.countryid, t1.countryname from countries as t1 join car_makers as t2 on t1.countryid = t2.country where t2.maker = "Fiat" group by t1.countryid having count(*) > 3 union select t1.countryid, t1.countryname from countries as t1 join model_list as t2 on t1.countryid = t2.modelid where t2.maker = "Fiat"	car_1
select country from airlines where airline = "JetBlue Airways"	flight_2
select country from airlines where airline = 'JetBlue Airways'	flight_2
select abbreviation from airlines where airline = "JetBlue Airways"	flight_2
select abbreviation from airlines where airline = "JetBlue Airways"	flight_2
select airline, abbreviation from airlines where country = 'USA'	flight_2
select airline, abbreviation from airlines where country = 'USA'	flight_2
select airportcode, airportname from airports where city = 'Anthony'	flight_2
select airportcode, airportname from airports where city = 'Anthony'	flight_2
select count(*) from airlines	flight_2
select count(*) from airlines	flight_2
select count(*) from airports	flight_2
select count(*) from airports	flight_2
select count(*) from flights	flight_2
select count(*) from flights	flight_2
select airline from airlines where abbreviation = 'UAL'	flight_2
select airline from airlines where abbreviation = 'UAL'	flight_2
select count(*) from airlines where country = 'USA'	flight_2
select count(*) from airlines where country = 'USA'	flight_2
select city, country from airports where airportname = "Alton"	flight_2
select city, country from airports where airportname = "Alton"	flight_2
select airportname from airports where airportcode = 'AKO'	flight_2
select airportname from airports where airportcode = 'AKO'	flight_2
select airportname from airports where city = 'Aberdeen'	flight_2
select airportname from airports where city = 'Aberdeen'	flight_2
select count(*) from flights where destairport = 'APG'	flight_2
select count(*) from flights where destairport = 'APG'	flight_2
select count(*) from airports as t1 join flights as t2 on t1.airportcode = t2.destairport where t1.airportcode = 'ATO'	flight_2
select count(*) from airports as t1 join flights as t2 on t1.airportcode = t2.sourceairport where t1.airportcode = 'ATO'	flight_2
select count(*) from airports as t1 join flights as t2 on t1.airportcode = t2.destairport where t1.city = 'Aberdeen'	flight_2
select count(*) from airports as t1 join flights as t2 on t1.airportcode = t2.destairport where t1.city = 'Aberdeen'	flight_2
select count(*) from airports as t1 join flights as t2 on t1.airportcode = t2.destairport where t1.city = 'Aberdeen'	flight_2
select count(*) from airports as t1 join flights as t2 on t1.airportcode = t2.destairport where t1.city = 'Aberdeen'	flight_2
select count(*) from flights as t1 join airports as t2 on t1.sourceairport = t2.airportcode where t2.city = 'Aberdeen' and t2.city = 'Ashley'	flight_2
select count(*) from flights as t1 join airports as t2 on t1.sourceairport = t2.airportcode and t1.destairport = t2.airportcode where t2.city = 'Aberdeen' and t2.airportname = 'Ashley'	flight_2
select count(*) from airlines as t1 join flights as t2 on t1.uid = t2.airline where t1.airline = 'JetBlue Airways'	flight_2
select count(*) from airlines as t1 join flights as t2 on t1.uid = t2.airline where t1.airline = "JetBlue Airways"	flight_2
select count(*) from airlines as t1 join flights as t2 on t1.uid = t2.airline where t1.airline = 'United Airlines' and t2.sourceairport = 'ASY'	flight_2
select count(*) from airlines as t1 join flights as t2 on t1.uid = t2.airline where t1.airline = 'United Airlines' and t2.airport	flight_2
select count(*) from airlines as t1 join flights as t2 on t1.uid = t2.airline where t1.airline = 'United Airlines' and t2.destairport = 'AHD'	flight_2
select count(*) from airlines as t1 join flights as t2 on t1.uid = t2.airline where t1.airline = 'United Airlines' and t2.sourceairport = 'AHD'	flight_2
select count(*) from flights as t1 join airports as t2 on t1.sourceairport = t2.airportcode join airlines as t3 on t1.airline = t3.uid where t2.city = 'Aberdeen' and t3.airline = 'United Airlines'	flight_2
select count(*) from airlines as t1 join flights as t2 on t1.uid = t2.airline where t1.country = 'United States' and t2.destairport = 'Aberdeen'	flight_2
select t1.city from airports as t1 join flights as t2 on t1.airportcode = t2.destairport group by t2.destairport order by count(*) desc limit 1	flight_2
select t1.city from airports as t1 join flights as t2 on t1.airportcode = t2.destairport group by t2.destairport order by count(*) desc limit 1	flight_2
select t1.city from airports as t1 join flights as t2 on t1.airportcode = t2.destairport group by t1.city order by count(*) desc limit 1	flight_2
select t3.city from flights as t1 join airports as t2 on t1.sourceairport = t2.airportcode join airports as t3 on t1.destairport = t3.airportcode group by t1.sourceairport order by count(*) desc limit 1	flight_2
select t1.airportcode from airports as t1 join flights as t2 on t1.airportcode = t2.sourceairport group by t1.airportcode order by count(*) desc limit 1	flight_2
select t1.airportcode from airports as t1 join flights as t2 on t1.airportcode = t2.sourceairport group by t1.airportcode order by count(*) desc limit 1	flight_2
select t1.airportcode from airports as t1 join flights as t2 on t1.airportcode = t2.sourceairport group by t2.sourceairport order by count(*) limit 1	flight_2
select t1.airportcode from airports as t1 join flights as t2 on t1.airportcode = t2.destairport group by t2.destairport order by count(*) limit 1	flight_2
select t1.airline from airlines as t1 join flights as t2 on t1.uid = t2.airline group by t2.airline order by count(*) desc limit 1	flight_2
select t1.airline from airlines as t1 join flights as t2 on t1.uid = t2.airline group by t2.airline order by count(*) desc limit 1	flight_2
select t1.abbreviation, t1.country from airlines as t1 join flights as t2 on t1.uid = t2.airline group by t2.airline order by count(*) limit 1	flight_2
select t1.abbreviation, t1.country from airlines as t1 join flights as t2 on t1.uid = t2.sourceairport group by t2.sourceairport order by count(*) limit 1	flight_2
select t1.airline from airlines as t1 join flights as t2 on t1.uid = t2.airline where t2.destairport = 'AHD'	flight_2
select t1.airline from airlines as t1 join flights as t2 on t1.uid = t2.airline where t2.sourceairport = 'AHD'	flight_2
select t1.airline from airlines as t1 join flights as t2 on t1.uid = t2.airline where t2.destairport = 'AHD'	flight_2
select t1.airline from airlines as t1 join flights as t2 on t1.uid = t2.airline where t2.destairport = 'AHD'	flight_2
select t1.airline from airlines as t1 join flights as t2 on t1.uid = t2.airline where t2.sourceairport = 'APG' intersect select t1.airline from airlines as t1 join flights as t2 on t1.uid = t2.airline where t2.destairport = 'CVO'	flight_2
select t1.airline from airlines as t1 join flights as t2 on t1.uid = t2.airline where t2.sourceairport = 'APG' intersect select t1.airline from airlines as t1 join flights as t2 on t1.uid = t2.airline where t2.destairport = 'CVO'	flight_2
select t1.airline from airlines as t1 join flights as t2 on t1.uid = t2.airline where t2.sourceairport = 'CVO' except select t1.airline from airlines as t1 join flights as t2 on t1.uid = t2.airline where t2.sourceairport = 'APG'	flight_2
select t1.airline from airlines as t1 join flights as t2 on t1.uid = t2.airline where t2.sourceairport = 'CVO' except select t1.airline from airlines as t1 join flights as t2 on t1.uid = t2.airline where t2.destairport = 'APG'	flight_2
select t1.airline from airlines as t1 join flights as t2 on t1.uid = t2.airline group by t1.uid having count(*) >= 10	flight_2
select t1.airline from airlines as t1 join flights as t2 on t1.uid = t2.airline group by t1.uid having count(*) >= 10	flight_2
select airline from flights group by airline having count(*) < 200	flight_2
select t1.airline from airlines as t1 join flights as t2 on t1.uid = t2.airline group by t1.uid having count(*) < 200	flight_2
select t1.flightno from flights as t1 join airlines as t2 on t1.airline = t2.uid where t2.airline = "United Airlines"	flight_2
select t1.flightno from flights as t1 join airlines as t2 on t1.airline = t2.uid where t2.airline = 'United Airlines'	flight_2
select flightno from flights where destairport = "APG"	flight_2
select flightno from flights where destairport = 'APG'	flight_2
select t1.flightno from flights as t1 join airports as t2 on t1.destairport = t2.airportcode where t2.airportcode = "APG"	flight_2
select t1.flightno from flights as t1 join airports as t2 on t1.destairport = t2.airportcode where t2.airportcode = 'APG'	flight_2
select t1.flightno from flights as t1 join airports as t2 on t1.destairport = t2.airportcode where t2.city = "Aberdeen"	flight_2
select t1.flightno from flights as t1 join airports as t2 on t1.sourceairport = t2.airportcode where t2.city = 'Aberdeen'	flight_2
select t1.flightno from flights as t1 join airports as t2 on t1.destairport = t2.airportcode where t2.city = "Aberdeen"	flight_2
select t1.flightno from flights as t1 join airports as t2 on t1.destairport = t2.airportcode where t2.city = 'Aberdeen'	flight_2
select count(*) from airports as t1 join flights as t2 on t1.airportcode = t2.destairport where t1.city = 'Aberdeen' or t1.city = 'Abilene'	flight_2
select count(*) from airports as t1 join flights as t2 on t1.airportcode = t2.destairport where t1.city = 'Aberdeen' or t1.city = 'Abilene'	flight_2
select airportname from airports where airportcode not in (select sourceairport, destairport from flights)	flight_2
select airportname from airports except select t1.airportname from airports as t1 join flights as t2 on t1.airportcode = t2.destairport	flight_2
select count(*) from employee	employee_hire_evaluation
select count(*) from employee	employee_hire_evaluation
select name from employee order by age asc	employee_hire_evaluation
select name from employee order by age asc	employee_hire_evaluation
select city, count(*) from employee group by city	employee_hire_evaluation
select city, count(*) from employee group by city	employee_hire_evaluation
select city from employee where age < 30 group by city having count(*) > 1	employee_hire_evaluation
select city from employee where age < 30 group by city having count(*) > 1	employee_hire_evaluation
select location, count(*) from shop group by location	employee_hire_evaluation
select location, count(*) from shop group by location	employee_hire_evaluation
select manager_name, district from shop order by number_products desc limit 1	employee_hire_evaluation
select manager_name, district from shop order by number_products desc limit 1	employee_hire_evaluation
select min(number_products), max(number_products) from shop	employee_hire_evaluation
select min(number_products), max(number_products) from shop	employee_hire_evaluation
select name, location, district from shop order by number_products desc	employee_hire_evaluation
select name, location, district from shop order by number_products desc	employee_hire_evaluation
select name from shop where number_products > (select avg(number_products) from shop)	employee_hire_evaluation
select name from shop where number_products > (select avg(number_products) from shop)	employee_hire_evaluation
select t1.name from employee as t1 join evaluation as t2 on t1.employee_id = t2.employee_id group by t2.employee_id order by count(*) desc limit 1	employee_hire_evaluation
select t1.name from employee as t1 join evaluation as t2 on t1.employee_id = t2.employee_id group by t2.employee_id order by count(*) desc limit 1	employee_hire_evaluation
select t1.name from employee as t1 join evaluation as t2 on t1.employee_id = t2.employee_id order by t2.bonus desc limit 1	employee_hire_evaluation
select t1.name from employee as t1 join evaluation as t2 on t1.employee_id = t2.employee_id order by t2.bonus desc limit 1	employee_hire_evaluation
select name from employee where employee_id not in (select employee_id from evaluation)	employee_hire_evaluation
select name from employee where employee_id not in (select employee_id from evaluation)	employee_hire_evaluation
select t2.name from hiring as t1 join shop as t2 on t1.shop_id = t2.shop_id group by t1.shop_id order by count(*) desc limit 1	employee_hire_evaluation
select t2.name from hiring as t1 join shop as t2 on t1.shop_id = t2.shop_id group by t1.shop_id order by count(*) desc limit 1	employee_hire_evaluation
select name from shop where shop_id not in (select shop_id from hiring)	employee_hire_evaluation
select name from shop where shop_id not in (select shop_id from hiring)	employee_hire_evaluation
select count(*), t3.name from hiring as t1 join employee as t2 on t1.employee_id = t2.employee_id join shop as t3 on t1.shop_id = t3.shop_id group by t3.shop_id	employee_hire_evaluation
select count(*), t3.name from hiring as t1 join employee as t2 on t1.employee_id = t2.employee_id join shop as t3 on t1.shop_id = t3.shop_id group by t3.name	employee_hire_evaluation
select sum(bonus) from evaluation	employee_hire_evaluation
select sum(bonus) from evaluation	employee_hire_evaluation
select * from hiring	employee_hire_evaluation
select * from hiring	employee_hire_evaluation
select district from shop where number_products < 3000 intersect select district from shop where number_products > 10000	employee_hire_evaluation
select district from shop where number_products < 3000 intersect select district from shop where number_products > 10000	employee_hire_evaluation
select count(distinct location) from shop	employee_hire_evaluation
select count(distinct location) from shop	employee_hire_evaluation
select count(*) from documents	cre_Doc_Template_Mgt
select count(*) from documents	cre_Doc_Template_Mgt
select document_id, document_name, document_description from documents	cre_Doc_Template_Mgt
select document_id, document_name, document_description from documents	cre_Doc_Template_Mgt
select document_name, template_id from documents where document_description like '%w%'	cre_Doc_Template_Mgt
select document_name, template_id from documents where document_description like '%w%'	cre_Doc_Template_Mgt
select document_id, template_id, document_description from documents where document_name = "Robbin CV"	cre_Doc_Template_Mgt
select document_id, template_id, document_description from documents where document_name = "Robbin CV"	cre_Doc_Template_Mgt
select count(distinct template_id) from documents	cre_Doc_Template_Mgt
select count(distinct template_id) from documents	cre_Doc_Template_Mgt
select count(*) from documents as t1 join templates as t2 on t1.template_id = t2.template_id where t2.template_type_code = 'PPT'	cre_Doc_Template_Mgt
select count(*) from documents as t1 join templates as t2 on t1.template_id = t2.template_id where t2.template_type_code = "PPT"	cre_Doc_Template_Mgt
select template_id, count(*) from documents group by template_id	cre_Doc_Template_Mgt
select template_id, count(*) from documents group by template_id	cre_Doc_Template_Mgt
select t1.template_id, t1.template_type_code from templates as t1 join documents as t2 on t1.template_id = t2.template_id group by t1.template_id order by count(*) desc limit 1	cre_Doc_Template_Mgt
select t1.template_id, t1.template_type_code from templates as t1 join documents as t2 on t1.template_id = t2.template_id group by t1.template_id order by count(*) desc limit 1	cre_Doc_Template_Mgt
select template_id from documents group by template_id having count(*) > 1	cre_Doc_Template_Mgt
select template_id from documents group by template_id having count(*) > 1	cre_Doc_Template_Mgt
select template_id from templates except select template_id from documents	cre_Doc_Template_Mgt
select template_id from templates except select template_id from documents	cre_Doc_Template_Mgt
select count(*) from templates	cre_Doc_Template_Mgt
select count(*) from templates	cre_Doc_Template_Mgt
select template_id, version_number, template_type_code from templates	cre_Doc_Template_Mgt
select template_id, version_number, template_type_code from templates	cre_Doc_Template_Mgt
select distinct template_type_code from templates	cre_Doc_Template_Mgt
select distinct template_type_code from templates	cre_Doc_Template_Mgt
select template_id from templates where template_type_code = "PP" or template_type_code = "PPT"	cre_Doc_Template_Mgt
select template_id from templates where template_type_code = "PP" or template_type_code = "PPT"	cre_Doc_Template_Mgt
select count(*) from templates where template_type_code = "CV"	cre_Doc_Template_Mgt
select count(*) from templates where template_type_code = "CV"	cre_Doc_Template_Mgt
select version_number, template_type_code from templates where version_number > 5	cre_Doc_Template_Mgt
select version_number, template_type_code from templates where version_number > 5	cre_Doc_Template_Mgt
select template_type_code, count(*) from templates group by template_type_code	cre_Doc_Template_Mgt
select template_type_code, count(*) from templates group by template_type_code	cre_Doc_Template_Mgt
select template_type_code from templates group by template_type_code order by count(*) desc limit 1	cre_Doc_Template_Mgt
select template_type_code from templates group by template_type_code order by count(*) desc limit 1	cre_Doc_Template_Mgt
select template_type_code from templates group by template_type_code having count(*) < 3	cre_Doc_Template_Mgt
select template_type_code from templates group by template_type_code having count(*) < 3	cre_Doc_Template_Mgt
select min(version_number), min(template_type_code) from templates	cre_Doc_Template_Mgt
select min(version_number), template_type_code from templates	cre_Doc_Template_Mgt
select t2.template_type_code from documents as t1 join templates as t2 on t1.template_id = t2.template_id where t1.document_name = "Data base"	cre_Doc_Template_Mgt
select t1.template_type_code from templates as t1 join documents as t2 on t1.template_id = t2.template_id where t2.document_name = "Data base"	cre_Doc_Template_Mgt
select document_name from documents as t1 join templates as t2 on t1.template_id = t2.template_id where t2.template_type_code = "BK"	cre_Doc_Template_Mgt
select t1.document_name from documents as t1 join templates as t2 on t1.template_id = t2.template_id where t2.template_type_code = "BK"	cre_Doc_Template_Mgt
select t1.template_type_code, count(*) from templates as t1 join documents as t2 on t1.template_id = t2.template_id group by t1.template_type_code	cre_Doc_Template_Mgt
select t1.template_type_code, count(*) from templates as t1 join documents as t2 on t1.template_id = t2.template_id group by t1.template_type_code	cre_Doc_Template_Mgt
select template_type_code from templates group by template_type_code order by count(*) desc limit 1	cre_Doc_Template_Mgt
select template_type_code from templates group by template_type_code order by count(*) desc limit 1	cre_Doc_Template_Mgt
select template_type_code from ref_template_types except select template_type_code from templates	cre_Doc_Template_Mgt
select template_type_code from ref_template_types except select template_type_code from templates	cre_Doc_Template_Mgt
select template_type_code, template_type_description from ref_template_types	cre_Doc_Template_Mgt
select template_type_code, template_type_description from ref_template_types	cre_Doc_Template_Mgt
select template_type_description from ref_template_types where template_type_code = "AD"	cre_Doc_Template_Mgt
select template_type_description from ref_template_types where template_type_code = "AD"	cre_Doc_Template_Mgt
select template_type_code from ref_template_types where template_type_description = "Book"	cre_Doc_Template_Mgt
select template_type_code from ref_template_types where template_type_description = "Book"	cre_Doc_Template_Mgt
select distinct t1.template_type_description from ref_template_types as t1 join templates as t2 on t1.template_type_code = t2.template_type_code join documents as t3 on t2.template_id = t3.template_id	cre_Doc_Template_Mgt
select distinct t1.template_type_description from ref_template_types as t1 join templates as t2 on t1.template_type_code = t2.template_type_code join documents as t3 on t2.template_id = t3.template_id	cre_Doc_Template_Mgt
select t1.template_id from templates as t1 join ref_template_types as t2 on t1.template_type_code = t2.template_type_code where t2.template_type_description = "Presentation"	cre_Doc_Template_Mgt
select t1.template_id from templates as t1 join ref_template_types as t2 on t1.template_type_code = t2.template_type_code where t2.template_type_description = "Presentation"	cre_Doc_Template_Mgt
select count(*) from paragraphs	cre_Doc_Template_Mgt
select count(*) from paragraphs	cre_Doc_Template_Mgt
select count(*) from documents as t1 join paragraphs as t2 on t1.document_id = t2.document_id where t1.document_name = "Summer Show"	cre_Doc_Template_Mgt
select count(*) from paragraphs as t1 join documents as t2 on t1.document_id = t2.document_id where t2.document_name = "Summer Show"	cre_Doc_Template_Mgt
select other_details from paragraphs where paragraph_text = 'Korea'	cre_Doc_Template_Mgt
select other_details from paragraphs where paragraph_text = 'Korea'	cre_Doc_Template_Mgt
select t1.paragraph_id, t1.paragraph_text from paragraphs as t1 join documents as t2 on t1.document_id = t2.document_id where t2.document_name = 'Welcome to NY'	cre_Doc_Template_Mgt
select t1.paragraph_id, t1.paragraph_text from paragraphs as t1 join documents as t2 on t1.document_id = t2.document_id where t2.document_name = 'Welcome to NY'	cre_Doc_Template_Mgt
select t1.paragraph_text from paragraphs as t1 join documents as t2 on t1.document_id = t2.document_id where t2.document_name = "Customer reviews"	cre_Doc_Template_Mgt
select t1.paragraph_text from paragraphs as t1 join documents as t2 on t1.document_id = t2.document_id where t2.document_name = "Customer reviews"	cre_Doc_Template_Mgt
select document_id, count(*) from paragraphs group by document_id order by document_id	cre_Doc_Template_Mgt
select document_id, count(*) from paragraphs group by document_id order by document_id	cre_Doc_Template_Mgt
select t1.document_id, t1.document_name, count(*) from documents as t1 join paragraphs as t2 on t1.document_id = t2.document_id group by t1.document_id	cre_Doc_Template_Mgt
select t1.document_id, t1.document_name, count(*) from documents as t1 join paragraphs as t2 on t1.document_id = t2.document_id group by t1.document_id	cre_Doc_Template_Mgt
select document_id from paragraphs group by document_id having count(*) >= 2	cre_Doc_Template_Mgt
select document_id from paragraphs group by document_id having count(*) >= 2	cre_Doc_Template_Mgt
select t1.document_id, t1.document_name from documents as t1 join paragraphs as t2 on t1.document_id = t2.document_id group by t1.document_id order by count(*) desc limit 1	cre_Doc_Template_Mgt
select t1.document_id, t1.document_name from documents as t1 join paragraphs as t2 on t1.document_id = t2.document_id group by t1.document_id order by count(*) desc limit 1	cre_Doc_Template_Mgt
select document_id from paragraphs group by document_id order by count(*) asc limit 1	cre_Doc_Template_Mgt
select document_id from paragraphs group by document_id order by count(*) asc limit 1	cre_Doc_Template_Mgt
select document_id from paragraphs group by document_id having count(*) between 1 and 2	cre_Doc_Template_Mgt
select document_id from paragraphs group by document_id having count(*) between 1 and 2	cre_Doc_Template_Mgt
select document_id from paragraphs where paragraph_text = 'Brazil' intersect select document_id from paragraphs where paragraph_text = 'Ireland'	cre_Doc_Template_Mgt
select document_id from paragraphs where paragraph_text = 'Brazil' intersect select document_id from paragraphs where paragraph_text = 'Ireland'	cre_Doc_Template_Mgt
select count(*) from teacher	course_teach
select count(*) from teacher	course_teach
select name from teacher order by age asc	course_teach
select name from teacher order by age asc	course_teach
select age, hometown from teacher	course_teach
select age, hometown from teacher	course_teach
select name from teacher where hometown!= 'Little Lever Urban District'	course_teach
select name from teacher where hometown!= 'Little Lever Urban District'	course_teach
select name from teacher where age = 32 or age = 33	course_teach
select name from teacher where age = 32 or age = 33	course_teach
select hometown from teacher order by age asc limit 1	course_teach
select hometown from teacher order by age asc limit 1	course_teach
select hometown, count(*) from teacher group by hometown	course_teach
select hometown, count(*) from teacher group by hometown	course_teach
select hometown from teacher group by hometown order by count(*) desc limit 1	course_teach
select hometown from teacher group by hometown order by count(*) desc limit 1	course_teach
select hometown from teacher group by hometown having count(*) >= 2	course_teach
select hometown from teacher group by hometown having count(*) >= 2	course_teach
select t2.name, t3.course from course_arrange as t1 join teacher as t2 on t1.teacher_id = t2.teacher_id join course as t3 on t1.course_id = t3.course_id	course_teach
select t2.name, t3.course from course_arrange as t1 join teacher as t2 on t1.teacher_id = t2.teacher_id join course as t3 on t1.course_id = t3.course_id	course_teach
select t2.name, t3.course from course_arrange as t1 join teacher as t2 on t1.teacher_id = t2.teacher_id join course as t3 on t1.course_id = t3.course_id order by t2.name asc	course_teach
select t2.name, t3.course from course_arrange as t1 join teacher as t2 on t1.teacher_id = t2.teacher_id join course as t3 on t1.course_id = t3.course_id order by t2.name asc	course_teach
select t2.name from course_arrange as t1 join teacher as t2 on t1.teacher_id = t2.teacher_id join course as t3 on t1.course_id = t3.course_id where t3.course = 'Math'	course_teach
select t2.name from course_arrange as t1 join teacher as t2 on t1.teacher_id = t2.teacher_id join course as t3 on t1.course_id = t3.course_id where t3.course = 'Math'	course_teach
select t2.name, count(*) from course_arrange as t1 join teacher as t2 on t1.teacher_id = t2.teacher_id group by t2.name	course_teach
select t2.name, count(*) from course_arrange as t1 join teacher as t2 on t1.teacher_id = t2.teacher_id group by t1.teacher_id	course_teach
select t2.name from course_arrange as t1 join teacher as t2 on t1.teacher_id = t2.teacher_id group by t1.teacher_id having count(*) >= 2	course_teach
select t2.name from course_arrange as t1 join teacher as t2 on t1.teacher_id = t2.teacher_id group by t1.teacher_id having count(*) >= 2	course_teach
select name from teacher where teacher_id not in (select teacher_id from course_arrange)	course_teach
select name from teacher where teacher_id not in (select teacher_id from course_arrange)	course_teach
select count(*) from visitor where age < 30	museum_visit
select name from visitor where level_of_membership > 4 order by level_of_membership desc	museum_visit
select avg(age) from visitor where level_of_membership <= 4	museum_visit
select name, level_of_membership from visitor where level_of_membership > 4 order by age desc	museum_visit
select museum_id, name from museum order by num_of_staff desc limit 1	museum_visit
select avg(num_of_staff) from museum where open_year < 2009	museum_visit
select open_year, num_of_staff from museum where name = "Plaza Museum"	museum_visit
select name from museum where num_of_staff > (select min(num_of_staff) from museum where open_year > 2010)	museum_visit
select t1.id, t1.name, t1.age from visitor as t1 join visit as t2 on t1.id = t2.visitor_id group by t2.visitor_id having count(*) > 1	museum_visit
select t1.id, t1.name, t1.level_of_membership from visitor as t1 join visit as t2 on t1.id = t2.visitor_id group by t1.id order by sum(total_spent) desc limit 1	museum_visit
select t1.museum_id, t1.name from museum as t1 join visit as t2 on t1.museum_id = t2.museum_id group by t2.museum_id order by count(*) desc limit 1	museum_visit
select name from museum where museum_id not in (select museum_id from visit)	museum_visit
select t1.name, t1.age from visitor as t1 join visit as t2 on t1.id = t2.visitor_id group by t2.visitor_id order by sum(t2.num_of_ticket) desc limit 1	museum_visit
select avg(num_of_ticket), max(num_of_ticket) from visit	museum_visit
select sum(t1.total_spent) from visit as t1 join visitor as t2 on t1.visitor_id = t2.id where t2.level_of_membership = 1	museum_visit
select t2.name from visit as t1 join visitor as t2 on t1.visitor_id = t2.id join museum as t3 on t1.museum_id = t3.museum_id where t3.open_year < 2009 intersect select t2.name from visit as t1 join visitor as t2 on t1.visitor_id = t2.id join museum as t3 on t1.museum_id = t3.museum_id where t3.open_year > 2011	museum_visit
select count(*) from visitor where id not in (select visitor_id from visit where museum_id in (select museum_id from museum where open_year > 2010))	museum_visit
select count(*) from museum where open_year > 2013 or open_year < 2008	museum_visit
select count(*) from players	wta_1
select count(*) from players	wta_1
select count(*) from matches	wta_1
select count(*) from matches	wta_1
select first_name, birth_date from players where country_code = 'USA'	wta_1
select first_name, birth_date from players where country_code = 'USA'	wta_1
select avg(loser_age), avg(winner_age) from matches	wta_1
select avg(loser_age), avg(winner_age) from matches	wta_1
select avg(winner_rank) from matches	wta_1
select avg(winner_rank) from matches	wta_1
select min(loser_rank) from matches	wta_1
select loser_rank from matches group by loser_rank order by sum(loser_rank_points) desc limit 1	wta_1
select count(distinct country_code) from players	wta_1
select count(distinct country_code) from players	wta_1
select count(distinct loser_name) from matches	wta_1
select count(distinct loser_name) from matches	wta_1
select tourney_name from matches group by tourney_name having count(*) > 10	wta_1
select tourney_name from matches group by tourney_name having count(*) > 10	wta_1
select winner_name from matches where year = 2013 intersect select winner_name from matches where year = 2016	wta_1
select winner_name from matches where year = 2013 intersect select winner_name from matches where year = 2016	wta_1
select count(*) from matches where year = 2013 or year = 2016	wta_1
select count(*) from matches where year = 2013 or year = 2016	wta_1
select t1.country_code, t1.first_name, t2.winner_name from players as t1 join matches as t2 on t1.player_id = t2.winner_id where t2.tourney_name = "WTA Championships" intersect select t1.country_code, t1.first_name from players as t1 join matches as t2 on t1.player_id = t2.winner_id where t2.tourney_name = "Australian Open"	wta_1
select t2.first_name, t2.country_code from matches as t1 join players as t2 on t1.winner_id = t2.player_id where t1.tourney_name = "WTA Championships" intersect select t2.first_name, t2.country_code from matches as t1 join players as t2 on t1.winner_id = t2.player_id where t1.t	wta_1
select first_name, country_code from players order by birth_date desc limit 1	wta_1
select first_name, country_code from players order by birth_date desc limit 1	wta_1
select first_name, last_name from players order by birth_date	wta_1
select first_name, last_name from players order by birth_date	wta_1
select first_name, last_name from players where hand = "left" order by birth_date	wta_1
select first_name, last_name from players where hand = "left" order by birth_date	wta_1
select t1.first_name, t1.country_code from players as t1 join rankings as t2 on t1.player_id = t2.player_id group by t2.player_id order by count(*) desc limit 1	wta_1
select t1.first_name, t1.country_code from players as t1 join rankings as t2 on t1.player_id = t2.player_id group by t2.player_id order by count(*) desc limit 1	wta_1
select year from matches group by year order by count(*) desc limit 1	wta_1
select year from matches group by year order by count(*) desc limit 1	wta_1
select t1.winner_name, t1.winner_rank_points from matches as t1 join rankings as t2 on t1.winner_id = t2.player_id group by t1.winner_id order by count(*) desc limit 1	wta_1
select t1.winner_name, t1.winner_rank_points from matches as t1 join rankings as t2 on t1.winner_id = t2.player_id group by t1.winner_name order by count(*) desc limit 1	wta_1
select winner_name from matches where tourney_name = "Australian Open" order by winner_rank_points desc limit 1	wta_1
select winner_name from matches where tourney_name = "Australian Open" order by winner_rank_points desc limit 1	wta_1
select loser_name, winner_name from matches order by minutes desc limit 1	wta_1
select t1.winner_name, t1.loser_name from matches as t1 join players as t2 on t1.winner_id = t2.player_id order by t1.minutes desc limit 1	wta_1
select avg(ranking), t1.first_name from players as t1 join rankings as t2 on t1.player_id = t2.player_id group by t1.player_id	wta_1
select t1.first_name, avg(t2.ranking) from players as t1 join rankings as t2 on t1.player_id = t2.player_id group by t1.player_id	wta_1
select sum(ranking_points), t1.first_name from players as t1 join rankings as t2 on t1.player_id = t2.player_id group by t1.player_id	wta_1
select t1.first_name, sum(t2.ranking_points) from players as t1 join rankings as t2 on t1.player_id = t2.player_id group by t1.player_id	wta_1
select country_code, count(*) from players group by country_code	wta_1
select country_code, count(*) from players group by country_code	wta_1
select country_code from players group by country_code order by count(*) desc limit 1	wta_1
select country_code from players group by country_code order by count(*) desc limit 1	wta_1
select country_code from players group by country_code having count(*) > 50	wta_1
select country_code from players group by country_code having count(*) > 50	wta_1
select ranking_date, count(*) from rankings group by ranking_date	wta_1
select ranking_date, sum(tours) from rankings group by ranking_date	wta_1
select count(*), year from matches group by year	wta_1
select year, count(*) from matches group by year	wta_1
select winner_name, winner_rank from matches order by winner_age limit 3	wta_1
select winner_name, winner_rank from matches order by winner_age limit 3	wta_1
select count(distinct winner_name) from matches where tourney_name = "WTA Championships" and winner_hand = "left"	wta_1
select count(*) from matches as t1 join players as t2 on t1.winner_id = t2.player_id where t1.tourney_name = "WTA Championships" and t2.hand = "left"	wta_1
select t1.first_name, t1.country_code, t1.birth_date from players as t1 join matches as t2 on t1.player_id = t2.winner_id order by t2.winner_rank_points desc limit 1	wta_1
select t1.first_name, t1.country_code, t1.birth_date from players as t1 join matches as t2 on t1.player_id = t2.winner_id group by t2.winner_id order by sum(t2.winner_rank_points) desc limit 1	wta_1
select hand, count(*) from players group by hand	wta_1
select hand, count(*) from players group by hand	wta_1
select count(*) from ship where disposition_of_ship = 'Captured'	battle_death
select name, tonnage from ship order by name desc	battle_death
select name, date, result from battle	battle_death
select max(killed), min(killed), t1.id from death as t1 join battle as t2 on t1.id = t2.id group by t1.id	battle_death
select avg(injured) from death	battle_death
select t1.killed, t1.injured from death as t1 join ship as t2 on t1.caused_by_ship_id = t2.id where t2.tonnage = 't'	battle_death
select name, result from battle where bulgarian_commander!= 'Boril'	battle_death
select distinct t1.id, t1.name from battle as t1 join ship as t2 on t1.id = t2.id where t2.ship_type = 'Brig'	battle_death
select t1.id, t1.name from battle as t1 join death as t2 on t1.id = t2.caused_by_ship_id group by t1.id having sum(t2.killed) > 10	battle_death
select t1.id, t1.name from ship as t1 join death as t2 on t1.id = t2.caused_by_ship_id group by t1.id order by sum(injured) desc limit 1	battle_death
select distinct name from battle where bulgarian_commander = 'Kaloyan' and latin_commander = 'Baldwin I'	battle_death
select count(distinct result) from battle	battle_death
select count(*) from battle where id not in ( select lost_in_battle from ship where tonnage = '225' )	battle_death
select t1.name, t1.date from battle as t1 join ship as t2 on t1.id = t2.lost_in_battle where t2.name = 'Lettice' intersect select t1.name, t1.date from battle as t1 join ship as t2 on t1.id = t2.lost_in_battle where t2.name = 'HMS Atalanta'	battle_death
select name, result, bulgarian_commander from battle where id not in (select lost_in_battle from ship where location = 'English Channel')	battle_death
select note from death where note like '%east%'	battle_death
select address_id, line_1, line_2 from addresses where address_id like "%1" intersect select address_id, line_1 from addresses where address_id like "%2"	student_transcripts_tracking
select line_1, line_2 from addresses	student_transcripts_tracking
select count(*) from courses	student_transcripts_tracking
select count(*) from courses	student_transcripts_tracking
select course_description from courses where course_name ='math'	student_transcripts_tracking
select course_description from courses where course_name ='math'	student_transcripts_tracking
select zip_postcode from addresses where city = "Port Chelsea"	student_transcripts_tracking
select zip_postcode from addresses where city = "Port Chelsea"	student_transcripts_tracking
select t2.department_name, t1.department_id from degree_programs as t1 join departments as t2 on t1.department_id = t2.department_id group by t1.department_id order by count(*) desc limit 1	student_transcripts_tracking
select t2.department_name, t1.department_id from degree_programs as t1 join departments as t2 on t1.department_id = t2.department_id group by t1.department_id order by count(*) desc limit 1	student_transcripts_tracking
select count(distinct department_id) from degree_programs	student_transcripts_tracking
select count(distinct department_id) from degree_programs	student_transcripts_tracking
select count(distinct degree_summary_name) from degree_programs	student_transcripts_tracking
select count(distinct degree_program_id) from degree_programs	student_transcripts_tracking
select count(*) from degree_programs as t1 join departments as t2 on t1.department_id = t2.department_id where t2.department_name = 'ENGINEERING'	student_transcripts_tracking
select count(*) from degree_programs as t1 join departments as t2 on t1.department_id = t2.department_id where t2.department_name = 'ENGINEERING'	student_transcripts_tracking
select section_name, section_description from sections	student_transcripts_tracking
select section_name, section_description from sections	student_transcripts_tracking
select t1.course_name, t1.course_id from courses as t1 join sections as t2 on t1.course_id = t2.course_id group by t1.course_id having count(*) <= 2	student_transcripts_tracking
select t1.course_name, t2.course_id from courses as t1 join sections as t2 on t1.course_id = t2.course_id group by t2.course_id having count(*) < 2	student_transcripts_tracking
select section_name from sections order by section_name desc	student_transcripts_tracking
select section_name from sections order by section_name desc	student_transcripts_tracking
select t1.semester_name, t2.semester_id from semesters as t1 join student_enrolment as t2 on t1.semester_id = t2.semester_id group by t2.semester_id order by count(*) desc limit 1	student_transcripts_tracking
select t1.semester_name, t1.semester_id from semesters as t1 join student_enrolment as t2 on t1.semester_id = t2.semester_id group by t1.semester_id order by count(*) desc limit 1	student_transcripts_tracking
select department_description from departments where department_name like '%computer%'	student_transcripts_tracking
select department_description from departments where department_name like '%computer%'	student_transcripts_tracking
select t2.first_name, t2.middle_name, t2.last_name, t1.student_enrolment_id from student_enrolment as t1 join students as t2 on t1.student_id = t2.student_id group by t1.student_enrolment_id having count(*) = 2	student_transcripts_tracking
select t2.first_name, t2.middle_name, t2.last_name, t1.student_id from student_enrolment as t1 join students as t2 on t1.student_id = t2.student_id group by t1.student_id having count(*) = 2	student_transcripts_tracking
select t2.first_name, t2.middle_name, t2.last_name from student_enrolment as t1 join students as t2 on t1.student_id = t2.student_id join degree_programs as t3 on t1.degree_program_id = t3.degree_program_id where t3.degree_summary_name = "Bachelor"	student_transcripts_tracking
select t2.first_name, t2.middle_name, t2.last_name from student_enrolment as t1 join students as t2 on t1.student_id = t2.student_id where t1.degree_program_id = 'Bachelors'	student_transcripts_tracking
select degree_program_id from student_enrolment group by degree_program_id order by count(*) desc limit 1	student_transcripts_tracking
select t1.degree_summary_name from degree_programs as t1 join student_enrolment as t2 on t1.degree_program_id = t2.degree_program_id group by t1.degree_summary_name order by count(*) desc limit 1	student_transcripts_tracking
select t1.degree_program_id, t3.degree_summary_name from student_enrolment as t1 join degree_programs as t2 on t1.degree_program_id = t2.degree_program_id join departments as t3	student_transcripts_tracking
select t1.degree_program_id, t2.degree_summary_name from student_enrolment as t1 join degree_programs as t2 on t1.degree_program_id = t2.degree_program_id group by t1.degree_program_id order by count(*) desc limit 1	student_transcripts_tracking
select t1.first_name, t1.middle_name, t1.last_name, count(*), t1.student_id from students as t1 join student_enrolment as t2 on t1.student_id = t2.student_id group by t1.student_id order by count(*) desc limit 1	student_transcripts_tracking
select t2.first_name, t2.middle_name, t2.last_name, t1.student_id, count(*) from student_enrolment as t1 join students as t2 on t1.student_id = t2.student_id group by t1.student_id order by count(*) desc limit 1	student_transcripts_tracking
select semester_name from semesters where semester_id not in (select semester_id from student_enrolment)	student_transcripts_tracking
select semester_name from semesters except select t1.semester_name from semesters as t1 join student_enrolment as t2 on t1.semester_id = t2.semester_id	student_transcripts_tracking
select t1.course_name from courses as t1 join student_enrolment_courses as t2 on t1.course_id = t2.course_id	student_transcripts_tracking
select t1.course_name from courses as t1 join student_enrolment_courses as t2 on t1.course_id = t2.course_id	student_transcripts_tracking
select t1.course_name from courses as t1 join student_enrolment_courses as t2 on t1.course_id = t2.course_id group by t1.course_name order by count(*) desc limit 1	student_transcripts_tracking
select t1.course_name from courses as t1 join student_enrolment_courses as t2 on t1.course_id = t2.course_id group by t1.course_name order by count(*) desc limit 1	student_transcripts_tracking
select t1.last_name from students as t1 join student_enrolment as t2 on t1.current_address_id = t2.student_id	student_transcripts_tracking
select last_name from students where permanent_address_id in (select student_id from student_enrolment where degree_program_id = 'UNC')	student_transcripts_tracking
select t1.transcript_date, t1.transcript_id from transcripts as t1 join transcript_contents as t2 on t1.transcript_id = t2.transcript_id group by t1.transcript_id having count(*) >= 2	student_transcripts_tracking
select t1.transcript_date, t1.transcript_id from transcripts as t1 join transcript_contents as t2 on t1.transcript_id = t2.transcript_id group by t1.transcript_id having count(*) >= 2	student_transcripts_tracking
select cell_mobile_number from students where first_name = "Timmothy" and last_name = "Ward"	student_transcripts_tracking
select cell_mobile_number from students where first_name = "Timmothy" and last_name = "Ward"	student_transcripts_tracking
select first_name, middle_name, last_name from students order by date_first_registered asc limit 1	student_transcripts_tracking
select first_name, middle_name, last_name from students order by date_first_registered asc limit 1	student_transcripts_tracking
select first_name, middle_name, last_name from students order by date_first_registered limit 1	student_transcripts_tracking
select first_name, middle_name, last_name from students order by date_first_registered limit 1	student_transcripts_tracking
select t2.first_name from addresses as t1 join students as t2 on t1.address_id = t2.permanent_address_id	student_transcripts_tracking
select first_name from students where permanent_address_id!= current_address_id	student_transcripts_tracking
select t1.address_id, t1.line_1 from addresses as t1 join students as t2 on t1.address_id = t2.current_address_id group by t1.address_id order by count(*) desc limit 1	student_transcripts_tracking
select t1.address_id, t1.line_1, t1.line_2 from addresses as t1 join students as t2 on t1.address_id = t2.permanent_address_id group by t1.address_id order by count(*) desc limit 1	student_transcripts_tracking
select avg(transcript_date) from transcripts	student_transcripts_tracking
select avg(transcript_date) from transcripts	student_transcripts_tracking
select transcript_date, other_details from transcripts order by transcript_date asc limit 1	student_transcripts_tracking
select transcript_date, other_details from transcripts order by transcript_date asc limit 1	student_transcripts_tracking
select count(*) from transcripts	student_transcripts_tracking
select count(*) from transcripts	student_transcripts_tracking
select transcript_date from transcripts order by transcript_date desc limit 1	student_transcripts_tracking
select transcript_date from transcripts order by transcript_date desc limit 1	student_transcripts_tracking
select count(*), t1.student_enrolment_id from student_enrolment as t1 join transcript_contents as t2 on t1.student_enrolment_id = t2.student_course_id group by t1.student_enrolment_id order by count(*) desc limit 1	student_transcripts_tracking
select max(course_id), student_enrolment_id from student_enrolment_courses	student_transcripts_tracking
select transcript_date, transcript_id from transcripts group by transcript_id order by count(*) asc limit 1	student_transcripts_tracking
select transcript_date, transcript_id from transcripts group by transcript_id order by count(*) asc limit 1	student_transcripts_tracking
select semester_id from student_enrolment where degree_program_id = 'Master' intersect select semester_id from student_enrolment where degree_program_id = 'Bachelor'	student_transcripts_tracking
select semester_id from student_enrolment where degree_program_id = 'MA' intersect select semester_id from student_enrolment where degree_program_id = 'B'	student_transcripts_tracking
select count(distinct current_address_id) from students	student_transcripts_tracking
select distinct t1.address_id from addresses as t1 join students as t2 on t1.address_id = t2.permanent_address_id	student_transcripts_tracking
select other_student_details from students order by other_student_details desc	student_transcripts_tracking
select other_student_details from students order by other_student_details desc	student_transcripts_tracking
select section_description from sections where section_name like '%h%'	student_transcripts_tracking
select section_description from sections where section_name = "h"	student_transcripts_tracking
select t2.first_name from addresses as t1 join students as t2 on t1.address_id = t2.permanent_address_id where t1.country = "Haiti" or t2.cell_mobile_number = "09700166582"	student_transcripts_tracking
select t2.first_name from addresses as t1 join students as t2 on t1.address_id = t2.permanent_address_id where t1.country = "Haiti" or t2.cell_mobile_number = "09700166582"	student_transcripts_tracking
select title from cartoon order by title	tvshow
select title from cartoon order by title	tvshow
select title from cartoon where directed_by = "Ben Jones"	tvshow
select title from cartoon where directed_by = 'Ben Jones'	tvshow
select count(*) from cartoon where written_by = "Joseph Kuhr"	tvshow
select count(*) from cartoon where written_by = "Joseph Kuhr"	tvshow
select title, directed_by from cartoon order by original_air_date	tvshow
select title, directed_by from cartoon order by original_air_date	tvshow
select title from cartoon where directed_by = "Ben Jones" or directed_by = "Brandon Vietti"	tvshow
select title from cartoon where directed_by = "Ben Jones" or directed_by = "Brandon Vietti"	tvshow
select country, count(*) from tv_channel group by country order by count(*) desc limit 1	tvshow
select country, count(*) from tv_channel group by country order by count(*) desc limit 1	tvshow
select count(distinct series_name), content from tv_channel	tvshow
select count(distinct series_name), count(distinct content) from tv_channel	tvshow
select content from tv_channel where series_name = "Sky Radio"	tvshow
select content from tv_channel where series_name = "Sky Radio"	tvshow
select package_option from tv_channel where series_name = "Sky Radio"	tvshow
select package_option from tv_channel where series_name = "Sky Radio"	tvshow
select count(*) from tv_channel where language = "English"	tvshow
select count(*) from tv_channel where language = "English"	tvshow
select language, count(*) from tv_channel group by language order by count(*) asc limit 1	tvshow
select language, count(*) from tv_channel group by language order by count(*) asc limit 1	tvshow
select language, count(*) from tv_channel group by language	tvshow
select language, count(*) from tv_channel group by language	tvshow
select t1.series_name from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.title = "The rise of the blue beetle!"	tvshow
select t1.series_name from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.title = "The rise of the blue beetle!"	tvshow
select t1.title from cartoon as t1 join tv_channel as t2 on t1.channel = t2.id where t2.series_name = "Sky Radio"	tvshow
select t1.title from cartoon as t1 join tv_series as t2 on t1.id = t2.id join tv_channel as t3 on t2.id = t3.id where t3.series_name = "Sky Radio"	tvshow
select episode from tv_series order by rating	tvshow
select episode from tv_series order by rating	tvshow
select episode, rating from tv_series order by rating desc limit 3	tvshow
select episode, rating from tv_series order by rating desc limit 3	tvshow
select min(share), max(share) from tv_series	tvshow
select max(share), min(share) from tv_series	tvshow
select air_date from tv_series where episode = "A Love of a Lifetime"	tvshow
select air_date from tv_series where episode = "A Love of a Lifetime"	tvshow
select weekly_rank from tv_series where episode = "A Love of a Lifetime"	tvshow
select weekly_rank from tv_series where episode = "A Love of a Lifetime"	tvshow
select t1.series_name from tv_channel as t1 join tv_series as t2 on t1.id = t2.channel where t2.episode = "A Love of a Lifetime"	tvshow
select t1.series_name from tv_channel as t1 join tv_series as t2 on t1.id = t2.id where t2.episode = "A Love of a Lifetime"	tvshow
select t1.episode from tv_series as t1 join tv_channel as t2 on t1.channel = t2.id where t2.series_name = "Sky Radio"	tvshow
select t1.episode from tv_series as t1 join tv_channel as t2 on t1.id = t2.id where t2.series_name = "Sky Radio"	tvshow
select directed_by, count(*) from cartoon group by directed_by	tvshow
select directed_by, count(*) from cartoon group by directed_by	tvshow
select production_code, channel from cartoon order by original_air_date desc limit 1	tvshow
select production_code, channel from cartoon order by original_air_date desc limit 1	tvshow
select package_option, series_name from tv_channel where hight_definition_tv = 'High'	tvshow
select series_name, package_option from tv_channel where hight_definition_tv = 'High'	tvshow
select t1.country from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.written_by = "Todd Casey"	tvshow
select t1.country from tv_channel as t1 join cartoon as t2 on t1.id = t2.id where t2.written_by = "Todd Casey"	tvshow
select country from tv_channel except select t1.country from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.written_by = "Todd Casey"	tvshow
select country from tv_channel except select t1.country from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.written_by = "Todd Casey"	tvshow
select t1.series_name, t1.country from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.directed_by = "Ben Jones" intersect select t1.series_name, t1.country from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.directed_by = "Michael Chang"	tvshow
select t1.series_name, t1.country from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.directed_by = "Ben Jones" intersect select t1.series_name, t1.country from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.directed_by = "Michael Chang"	tvshow
select pixel_aspect_ratio_par, country from tv_channel where language!= "English"	tvshow
select pixel_aspect_ratio_par, country from tv_channel where language!= "English"	tvshow
select id from tv_channel where country = 2 group by country having count(*) > 2	tvshow
select id from tv_channel group by id having count(*) > 2	tvshow
select id from tv_channel except select channel from cartoon where directed_by = "Ben Jones"	tvshow
select id from tv_channel except select channel from cartoon where directed_by = "Ben Jones"	tvshow
select package_option from tv_channel except select t1.package_option from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.directed_by = "Ben Jones"	tvshow
select package_option from tv_channel except select t1.package_option from tv_channel as t1 join cartoon as t2 on t1.id = t2.channel where t2.directed_by = "Ben Jones"	tvshow
select count(*) from poker_player	poker_player
select count(*) from poker_player	poker_player
select earnings from poker_player order by earnings desc	poker_player
select earnings from poker_player order by earnings desc	poker_player
select final_table_made, best_finish from poker_player	poker_player
select final_table_made, best_finish from poker_player	poker_player
select avg(earnings) from poker_player	poker_player
select avg(earnings) from poker_player	poker_player
select money_rank from poker_player order by earnings desc limit 1	poker_player
select money_rank from poker_player order by earnings desc limit 1	poker_player
select max(final_table_made) from poker_player where earnings < 200000	poker_player
select max(final_table_made) from poker_player where earnings < 200000	poker_player
select t2.name from poker_player as t1 join people as t2 on t1.people_id = t2.people_id	poker_player
select t2.name from poker_player as t1 join people as t2 on t1.people_id = t2.people_id	poker_player
select t2.name from poker_player as t1 join people as t2 on t1.people_id = t2.people_id where t1.earnings > 300000	poker_player
select t2.name from poker_player as t1 join people as t2 on t1.people_id = t2.people_id where t1.earnings > 300000	poker_player
select t2.name from poker_player as t1 join people as t2 on t1.people_id = t2.people_id order by t1.final_table_made asc	poker_player
select t2.name from poker_player as t1 join people as t2 on t1.people_id = t2.people_id order by t1.final_table_made asc	poker_player
select t2.birth_date from poker_player as t1 join people as t2 on t1.people_id = t2.people_id order by t1.earnings limit 1	poker_player
select t2.birth_date from poker_player as t1 join people as t2 on t1.people_id = t2.people_id order by t1.earnings limit 1	poker_player
select t2.money_rank from people as t1 join poker_player as t2 on t1.people_id = t2.people_id order by t1.height desc limit 1	poker_player
select t2.money_rank from people as t1 join poker_player as t2 on t1.people_id = t2.people_id order by t1.height desc limit 1	poker_player
select avg(t1.earnings) from poker_player as t1 join people as t2 on t1.people_id = t2.people_id where t2.height > 200	poker_player
select avg(t1.earnings) from poker_player as t1 join people as t2 on t1.people_id = t2.people_id where t2.height > 200	poker_player
select t2.name from poker_player as t1 join people as t2 on t1.people_id = t2.people_id order by t1.earnings desc	poker_player
select t2.name from poker_player as t1 join people as t2 on t1.people_id = t2.people_id order by t1.earnings desc	poker_player
select nationality, count(*) from people group by nationality	poker_player
select nationality, count(*) from people group by nationality	poker_player
select nationality from people group by nationality order by count(*) desc limit 1	poker_player
select nationality from people group by nationality order by count(*) desc limit 1	poker_player
select nationality from people group by nationality having count(*) >= 2	poker_player
select nationality from people group by nationality having count(*) >= 2	poker_player
select name, birth_date from people order by name asc	poker_player
select name, birth_date from people order by name asc	poker_player
select name from people where nationality!= 'Russia'	poker_player
select name from people where nationality!= 'Russia'	poker_player
select name from people where people_id not in (select people_id from poker_player)	poker_player
select name from people where people_id not in (select people_id from poker_player)	poker_player
select count(distinct nationality) from people	poker_player
select count(distinct nationality) from people	poker_player
select count(distinct state) from area_code_state	voter_1
select contestant_number, contestant_name from contestants order by contestant_name desc	voter_1
select vote_id, phone_number, state from votes	voter_1
select max(area_code), min(area_code) from area_code_state	voter_1
select created from votes where state = 'CA' order by created desc limit 1	voter_1
select contestant_name from contestants where contestant_name!= 'Jessie Alloway'	voter_1
select distinct state, created from votes	voter_1
select t2.contestant_number, t2.contestant_name from votes as t1 join contestants as t2 on t1.contestant_number = t2.contestant_number group by t2.contestant_number having count(*) >= 2	voter_1
select t1.contestant_number, t1.contestant_name from contestants as t1 join votes as t2 on t1.contestant_number = t2.contestant_number group by t1.contestant_number order by count(*) asc limit 1	voter_1
select count(*) from votes where state = 'NY' or state = 'CA'	voter_1
select count(*) from contestants where contestant_number not in (select contestant_number from votes)	voter_1
select t1.area_code from area_code_state as t1 join votes as t2 on t1.area_code = t2.area_code	voter_1
select t1.created, t1.state, t1.phone_number from votes as t1 join contestants as t2 on t1.contestant_number = t2.contestant_number where t2.contestant_name = 'Tabatha Gehling'	voter_1
select t1.area_code from area_code_state as t1 join votes as t2 on t1.area_code = t2.vote_id join contestants as t3 on t2.contestant_number = t3.contestant_number where t3.contestant_name = 'Tabatha Gehling' intersect select t1.area_code from area_code_state as t1 join votes as t2 on t1.area_code = t2.area_code	voter_1
select contestant_name from contestants where contestant_name like '%al%'	voter_1
select name from country where indepyear > 1950	world_1
select name from country where indepyear > 1950	world_1
select count(*) from country where governmentform = "Republic"	world_1
select count(*) from country where governmentform = "Republic"	world_1
select sum(surfacearea) from country where region = 'Caribbean'	world_1
select sum(surfacearea) from country where region = 'Caribbean'	world_1
select continent from country where localname = "Anguilla"	world_1
select continent from country where name = "Anguilla"	world_1
select region from city as t1 join country as t2 on t1.countrycode = t2.code where t1.name = 'Kabul'	world_1
select region from city as t1 join country as t2 on t1.id = t2.code where t1.name = "Kabul"	world_1
select t2.language from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t1.name = "Aruba" group by t2.language order by count(*) desc limit 1	world_1
select language from country as t1 join city as t2 on t1.code = t2.countrycode join countrylanguage as t3 on t2.countrycode = t3.countrycode where t1.name = "Aruba" and t3.language = "Spanish" group by t3.language order by count(*) desc limit 1	world_1
select population, lifeexpectancy from country where name = "Brazil"	world_1
select population, lifeexpectancy from country where name = "Brazil"	world_1
select region, population from country where name = "Angola"	world_1
select region, population from country where name = "Angola"	world_1
select avg(lifeexpectancy) from country where region = "Central Africa"	world_1
select avg(lifeexpectancy) from country where region = "Central Africa"	world_1
select name from country where continent = 'Asia' order by lifeexpectancy asc limit 1	world_1
select name from country where continent = 'Asia' order by lifeexpectancy limit 1	world_1
select sum(population), max(gnp) from country where continent = 'Asia'	world_1
select population, gnp from country where continent = 'Asia' order by gnp desc limit 1	world_1
select avg(lifeexpectancy) from country where continent = 'Africa' and governmentform = 'Republic'	world_1
select avg(lifeexpectancy) from country where continent = 'Africa' and governmentform = 'Republic'	world_1
select sum(surfacearea) from country where continent = 'Asia' or continent = 'Europe'	world_1
select sum(surfacearea) from country where continent = 'Asia' or continent = 'Europe'	world_1
select population from city where district = "Gelderland"	world_1
select sum(population) from city where district = 'Gelderland'	world_1
select avg(gnp), sum(population) from country where governmentform = "US Territory"	world_1
select avg(gnp), sum(population) from country where governmentform = 'US Territory'	world_1
select count(distinct language) from countrylanguage	world_1
select count(distinct language) from countrylanguage	world_1
select count(distinct governmentform) from country where continent = 'Africa'	world_1
select count(distinct governmentform) from country where continent = 'Africa'	world_1
select count(distinct t1.language) from countrylanguage as t1 join city as t2 on t1.countrycode = t2.countrycode where t2.name = "Aruba"	world_1
select count(distinct t1.language) from countrylanguage as t1 join city as t2 on t1.countrycode = t2.countrycode where t2.name = "Aruba"	world_1
select count(distinct language) from countrylanguage as t1 join country as t2 on t1.countrycode = t2.code where t2.name = 'Afghanistan'	world_1
select count(distinct language) from countrylanguage as t1 join country as t2 on t1.countrycode = t2.code where t2.name = 'Afghanistan'	world_1
select t1.name from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode group by t1.name order by count(*) desc limit 1	world_1
select t1.name from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode group by t2.countrycode order by count(distinct language) desc limit 1	world_1
select t1.continent from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode group by t1.continent order by count(*) desc limit 1	world_1
select t1.continent from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode group by t1.continent order by count(*) desc limit 1	world_1
select count(*) from countrylanguage where language = "English" intersect select count(*) from countrylanguage where language = "Dutch"	world_1
select count(*) from countrylanguage where language = "English" intersect select count(*) from countrylanguage where language = "Dutch"	world_1
select t1.name from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t2.language = "English" intersect select t1.name from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t2.language = "French"	world_1
select t1.name from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t2.language = "English" intersect select t1.name from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t2.language = "French"	world_1
select t1.name from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t2.language = "English" intersect select t1.name from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t2.language = "French"	world_1
select t1.name from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t2.language = "English" intersect select t1.name from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t2.language = "French"	world_1
select count(distinct t1.continent) from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t2.language = "Chinese"	world_1
select count(*) from countrylanguage where language = "Chinese"	world_1
select distinct t1.region from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t2.language = "English" or t2.language = "Dutch"	world_1
select distinct t1.region from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t2.language = "Dutch" or t2.language = "English"	world_1
select countrycode from countrylanguage where language = "English" or language = "Dutch"	world_1
select countrycode from countrylanguage where language = "English" or language = "Dutch"	world_1
select t2.language from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t1.continent = 'Asia' group by t2.language order by count(*) desc limit 1	world_1
select t2.language from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t1.continent = 'Asia' group by t2.language order by count(*) desc limit 1	world_1
select distinct t2.language from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t1.governmentform = 'Republic' group by t2.language having count(*) = 1	world_1
select distinct t2.language from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t1.governmentform = "Republic"	world_1
select t1.name from city as t1 join countrylanguage as t2 on t1.countrycode = t2.countrycode where t2.language = "English" order by t1.population desc limit 1	world_1
select t1.name from city as t1 join countrylanguage as t2 on t1.countrycode = t2.countrycode where t2.language = "English" order by t1.population desc limit 1	world_1
select name, population, lifeexpectancy from country where continent = 'Asia' order by surfacearea desc limit 1	world_1
select name, population, lifeexpectancy from country where continent = 'Asia' order by surfacearea desc limit 1	world_1
select avg(t2.lifeexpectancy) from countrylanguage as t1 join country as t2 on t1.countrycode = t2.code where t1.language!= "English"	world_1
select avg(t2.lifeexpectancy) from countrylanguage as t1 join country as t2 on t1.countrycode = t2.code where t1.language!= "English"	world_1
select sum(population) from country where code not in (select countrycode from countrylanguage where language = 'English')	world_1
select sum(population) from country where countrycode not in (select countrycode from countrylanguage where language = 'English' or countrycode not in (select countrycode from countrylanguage where language = 'English' or countrycode not in (select countrycode from countrylanguage where language = 'English' or countrycode not in (select countrycode from countrylanguage where language = 'English' or countrycode not in (select countrycode from countrylanguage where	world_1
select t2.language from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t1.headofstate = "Beatrix"	world_1
select t2.language from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t1.name = "Beatrix"	world_1
select count(distinct t2.language) from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t1.indepyear < 1930	world_1
select count(distinct language) from countrylanguage where countrycode < 1930	world_1
select name from country where surfacearea > (select max(surfacearea) from country where continent = 'Europe')	world_1
select name from country where surfacearea > (select max(surfacearea) from country where continent = 'Europe')	world_1
select name from country where continent = 'Africa' and population < (select min(population) from country where continent = 'Asia')	world_1
select name from country where population < (select min(population) from country where continent = 'Asia')	world_1
select name from country where population > (select max(population) from country where continent = 'Africa')	world_1
select name from country where population > (select max(population) from country where continent = 'Africa')	world_1
select countrycode from countrylanguage where language!= "English"	world_1
select countrycode from countrylanguage where language!= "English"	world_1
select countrycode from countrylanguage where language!= "English"	world_1
select countrycode from countrylanguage where language!= "English"	world_1
	world_1
select code from country where governmentform!= 'Republic' and language!= 'English	world_1
select t1.name from city as t1 join country as t2 on t1.countrycode = t2.code where t2.continent = 'Europe' and t2.language	world_1
select name from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t1.continent = 'Europe' and t2.language!= "English"	world_1
select distinct t1.name from city as t1 join countrylanguage as t2 on t1.countrycode = t2.countrycode where t2.language = "Chinese"	world_1
select distinct t1.name from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode where t1.continent = 'Asia' and t2.language = 'Chinese'	world_1
select name, indepyear, surfacearea from country order by population asc limit 1	world_1
select name, indepyear, surfacearea from country order by population limit 1	world_1
select population, name, headofstate from country order by surfacearea desc limit 1	world_1
select name, population, headofstate from country order by surfacearea desc limit 1	world_1
select t1.name, count(distinct t2.language) from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode group by t1.code having count(distinct t2.language) >= 3	world_1
select t1.name, count(distinct t2.language) from country as t1 join countrylanguage as t2 on t1.code = t2.countrycode group by t1.code having count(distinct t2.language) > 2	world_1
select count(*), district from city where population > (select avg(population) from city) group by district	world_1
select count(*), district from city where population > (select avg(population) from city) group by district	world_1
select governmentform, sum(population) from country group by governmentform having avg(lifeexpectancy) > 72	world_1
select governmentform, sum(population) from country group by governmentform having avg(lifeexpectancy) > 72	world_1
select avg(lifeexpectancy), sum(population), continent from country where avg(lifeexpectancy) < 72 group by continent	world_1
select continent, sum(population), avg(lifeexpectancy) from country group by continent having avg(lifeexpectancy) < 72	world_1
select name, surfacearea from country order by surfacearea desc limit 5	world_1
select name, surfacearea from country order by surfacearea desc limit 5	world_1
select name from country order by population desc limit 3	world_1
select name from country order by population desc limit 3	world_1
select name from country order by population asc limit 3	world_1
select name from country order by population asc limit 3	world_1
select count(*) from country where continent = 'Asia'	world_1
select count(*) from country where continent = 'Asia'	world_1
select name from country where continent = 'Europe' and population = 80000	world_1
select name from country where continent = 'Europe' and population = 80000	world_1
select sum(population), avg(surfacearea) from country where continent = 'North America' and surfacearea > 3000	world_1
select sum(population), avg(surfacearea) from country where continent = 'North America' and surfacearea > 3000	world_1
select name from city where population between 160000 and 900000	world_1
select name from city where population between 160000 and 900000	world_1
select language from countrylanguage group by language order by count(*) desc limit 1	world_1
select language from countrylanguage group by language order by count(*) desc limit 1	world_1
select language, percentage from countrylanguage group by countrycode	world_1
select countrycode, language, percentage from countrylanguage group by countrycode order by percentage desc limit 1	world_1
select count(*) from countrylanguage where language = "Spanish" group by countrycode order by percentage desc limit 1	world_1
select count(*) from countrylanguage where language = "Spanish" and percentage >= 1	world_1
select countrycode from countrylanguage where language = "Spanish" group by countrycode order by percentage desc limit 1	world_1
select countrycode from countrylanguage where language = "Spanish" group by countrycode having count(*) >= 2	world_1
select count(*) from conductor	orchestra
select count(*) from conductor	orchestra
select name from conductor order by age asc	orchestra
select name from conductor order by age	orchestra
select name from conductor where nationality!= "USA"	orchestra
select name from conductor where nationality!= 'USA'	orchestra
select record_company from orchestra order by year_of_founded desc	orchestra
select record_company from orchestra order by year_of_founded desc	orchestra
select avg(attendance) from show	orchestra
select avg(attendance) from show	orchestra
select max(share), min(share) from performance where type!= "Live final"	orchestra
select max(share), min(share) from performance where type!= "Live final"	orchestra
select count(distinct nationality) from conductor	orchestra
select count(distinct nationality) from conductor	orchestra
select name from conductor order by year_of_work desc	orchestra
select name from conductor order by year_of_work desc	orchestra
select name from conductor order by year_of_work desc limit 1	orchestra
select name from conductor order by year_of_work desc limit 1	orchestra
select t1.name, t2.orchestra from conductor as t1 join orchestra as t2 on t1.conductor_id = t2.conductor_id	orchestra
select t1.name, t2.orchestra from conductor as t1 join orchestra as t2 on t1.conductor_id = t2.conductor_id	orchestra
select t2.name from orchestra as t1 join conductor as t2 on t1.conductor_id = t2.conductor_id group by t1.conductor_id having count(*) > 1	orchestra
select t1.name from conductor as t1 join orchestra as t2 on t1.conductor_id = t2.conductor_id group by t2.conductor_id having count(*) > 1	orchestra
select t2.name from orchestra as t1 join conductor as t2 on t1.conductor_id = t2.conductor_id group by t1.conductor_id order by count(*) desc limit 1	orchestra
select t2.name from orchestra as t1 join conductor as t2 on t1.conductor_id = t2.conductor_id group by t1.conductor_id order by count(*) desc limit 1	orchestra
select t2.name from orchestra as t1 join conductor as t2 on t1.conductor_id = t2.conductor_id where t1.year_of_founded > 2008	orchestra
select t2.name from orchestra as t1 join conductor as t2 on t1.conductor_id = t2.conductor_id where t1.year_of_founded > 2008	orchestra
select record_company, count(*) from orchestra group by record_company	orchestra
select record_company, count(*) from orchestra group by record_company	orchestra
select major_record_format from orchestra group by major_record_format order by count(*) asc	orchestra
select major_record_format from orchestra group by major_record_format order by count(*) asc	orchestra
select record_company from orchestra group by record_company order by count(*) desc limit 1	orchestra
select record_company from orchestra group by record_company order by count(*) desc limit 1	orchestra
select orchestra from orchestra where orchestra_id not in (select orchestra_id from performance)	orchestra
select orchestra from orchestra where orchestra_id not in (select orchestra_id from performance)	orchestra
select record_company from orchestra where year_of_founded < 2003 intersect select record_company from orchestra where year_of_founded > 2003	orchestra
select record_company from orchestra where year_of_founded < 2003 intersect select record_company from orchestra where year_of_founded > 2003	orchestra
select count(*) from orchestra where major_record_format = "CD" or major_record_format = "DVD"	orchestra
select count(*) from orchestra where major_record_format = "CD" or major_record_format = "DVD"	orchestra
select t1.year_of_founded from orchestra as t1 join performance as t2 on t1.orchestra_id = t2.orchestra_id group by t1.orchestra_id having count(*) > 1	orchestra
select t2.year_of_founded from performance as t1 join orchestra as t2 on t1.orchestra_id = t2.orchestra_id group by t1.orchestra_id having count(*) > 1	orchestra
select count(*) from highschooler	network_1
select count(*) from highschooler	network_1
select name, grade from highschooler	network_1
select name, grade from highschooler	network_1
select distinct grade from highschooler	network_1
select grade from highschooler	network_1
select grade from highschooler where name = 'Kyle'	network_1
select grade from highschooler where name = 'Kyle'	network_1
select name from highschooler where grade = 10	network_1
select name from highschooler where grade = 10	network_1
select id from highschooler where name = 'Kyle'	network_1
select id from highschooler where name = 'Kyle'	network_1
select count(*) from highschooler where grade = 9 or grade = 10	network_1
select count(*) from highschooler where grade = 9 or grade = 10	network_1
select count(*), grade from highschooler group by grade	network_1
select count(*), grade from highschooler group by grade	network_1
select grade from highschooler group by grade order by count(*) desc limit 1	network_1
select grade from highschooler group by grade order by count(*) desc limit 1	network_1
select grade from highschooler group by grade having count(*) >= 4	network_1
select grade from highschooler group by grade having count(*) >= 4	network_1
select student_id, count(*) from friend group by student_id	network_1
select count(*), t1.name from highschooler as t1 join friend as t2 on t1.id = t2.student_id group by t1.id	network_1
select t1.name, count(*) from highschooler as t1 join friend as t2 on t1.id = t2.student_id group by t1.id	network_1
select count(*), t1.name from highschooler as t1 join friend as t2 on t1.id = t2.friend_id group by t1.name	network_1
select t1.name from highschooler as t1 join friend as t2 on t1.id = t2.friend_id group by t1.id order by count(*) desc limit 1	network_1
select t1.name from highschooler as t1 join friend as t2 on t1.id = t2.student_id group by t2.student_id order by count(*) desc limit 1	network_1
select t1.name from highschooler as t1 join friend as t2 on t1.id = t2.friend_id group by t1.id having count(*) >= 3	network_1
select t1.name from highschooler as t1 join friend as t2 on t1.id = t2.friend_id group by t1.id having count(*) >= 3	network_1
select t3.name from highschooler as t1 join friend as t2 on t1.id = t2.friend_id join highschooler as t3 on t1.id = t3.id where t1.name = 'Kyle'	network_1
select t3.name from highschooler as t1 join friend as t2 on t1.id = t2.friend_id join highschooler as t3 on t1.id = t2.student_id where t1.name = 'Kyle'	network_1
select count(t2.friend_id) from highschooler as t1 join friend as t2 on t1.id = t2.student_id where t1.name = 'Kyle'	network_1
select count(*) from friend as t1 join highschooler as t2 on t1.student_id = t2.id where t2.name = 'Kyle'	network_1
select id from highschooler except select student_id from friend	network_1
select id from highschooler except select student_id from friend	network_1
select name from highschooler where id not in (select student_id from friend)	network_1
select name from highschooler where id not in (select student_id from friend)	network_1
select t1.id from highschooler as t1 join friend as t2 on t1.id = t2.student_id intersect select t1.id from highschooler as t1 join likes as t2 on t1.id = t2.student_id	network_1
select student_id from friend intersect select student_id from likes	network_1
select t1.name from highschooler as t1 join friend as t2 on t1.id = t2.student_id intersect select t1.name from highschooler as t1 join likes as t2 on t1.id = t2.student_id	network_1
select t1.name from highschooler as t1 join friend as t2 on t1.id = t2.student_id intersect select t1.name from highschooler as t1 join likes as t2 on t1.id = t2.student_id	network_1
select count(*), student_id from likes group by student_id	network_1
select student_id, count(*) from likes group by student_id	network_1
select t1.name, count(*) from highschooler as t1 join likes as t2 on t1.id = t2.student_id group by t1.id	network_1
select t1.name, count(*) from highschooler as t1 join likes as t2 on t1.id = t2.student_id group by t1.id	network_1
select t1.name from highschooler as t1 join likes as t2 on t1.id = t2.student_id group by t2.student_id order by count(*) desc limit 1	network_1
select t1.name from highschooler as t1 join likes as t2 on t1.id = t2.student_id group by t2.student_id order by count(*) desc limit 1	network_1
select t1.name from highschooler as t1 join likes as t2 on t1.id = t2.student_id group by t2.student_id having count(*) >= 2	network_1
select t1.name from highschooler as t1 join likes as t2 on t1.id = t2.student_id group by t2.student_id having count(*) >= 2	network_1
select t1.name from highschooler as t1 join friend as t2 on t1.id = t2.student_id where t1.grade > 5 group by t2.student_id having count(*) >= 2	network_1
select t1.name from highschooler as t1 join friend as t2 on t1.id = t2.friend_id where t1.grade > 5 group by t1.id having count(*) >= 2	network_1
select count(*) from highschooler as t1 join likes as t2 on t1.id = t2.student_id where t1.name = 'Kyle'	network_1
select count(*) from likes as t1 join highschooler as t2 on t1.student_id = t2.id where t2.name = "Kyle"	network_1
select avg(t2.grade) from friend as t1 join highschooler as t2 on t1.student_id = t2.id	network_1
select avg(grade) from highschooler join friend	network_1
select min(grade) from highschooler where id not in (select student_id from friend)	network_1
select min(grade) from highschooler where id not in (select student_id from friend)	network_1
select state from owners intersect select state from professionals	dog_kennels
select state from owners intersect select state from professionals	dog_kennels
select avg(t2.age) from treatments as t1 join dogs as t2 on t1.dog_id = t2.dog_id	dog_kennels
select avg(t2.age) from treatments as t1 join dogs as t2 on t1.dog_id = t2.dog_id	dog_kennels
select t2.professional_id, t2.first_name, t2.last_name, t2.cell_number from treatments as t1 join professionals as t2 on t1.professional_id = t2.professional_id where t2.state = "Indiana" group by t2.professional_id having count(*) > 2	dog_kennels
select t1.professional_id, t1.last_name, t1.cell_number from professionals as t1 join treatments as t2 on t1.professional_id = t2.professional_id group by t1.professional_id having t1.state = "Indiana" union select t1.professional_id, t1.last_name, t1.cell_number from professionals as t1 join treatments as t2 on t1.professional_id = t2.professional_id group by t1.professional_id having count(*) > 2	dog_kennels
select name from dogs where owner_id not in (select dog_id from treatments where cost_of_treatment > 1000)	dog_kennels
select name from dogs where owner_id not in (select dog_id from treatments group by dog_id having sum(cost_of_treatment) > 1000)	dog_kennels
select first_name from professionals union select first_name from owners except select t2.name from professionals as t1 join dogs as t2 on t1.professional_id = t2.owner_id	dog_kennels
select first_name from professionals union select first_name from owners except select t2.name from professionals as t1 join dogs as t2 on t1.professional_id = t2.owner_id	dog_kennels
select professional_id, role_code, email_address from professionals except select t2.professional_id, t2.role_code, t2.email_address from treatments as t1 join professionals as t2 on t1.professional_id = t2.professional_id	dog_kennels
select professional_id, role_code, email_address from professionals except select t1.professional_id, t1.role_code, t1.email_address from professionals as t1 join treatments as t2 on t1.professional_id = t2.professional_id	dog_kennels
select t1.owner_id, t1.first_name, t1.last_name from owners as t1 join dogs as t2 on t1.owner_id = t2.owner_id group by t1.owner_id order by count(*) desc limit 1	dog_kennels
select t1.owner_id, t1.first_name, t1.last_name from owners as t1 join dogs as t2 on t1.owner_id = t2.owner_id group by t1.owner_id order by count(*) desc limit 1	dog_kennels
select t1.professional_id, t1.role_code, t1.first_name from professionals as t1 join treatments as t2 on t1.professional_id = t2.professional_id group by t1.professional_id having count(*) >= 2	dog_kennels
select t1.professional_id, t1.role_code, t1.first_name from professionals as t1 join treatments as t2 on t1.professional_id = t2.professional_id group by t1.professional_id having count(*) >= 2	dog_kennels
select breed_name from breeds group by breed_code order by count(*) desc limit 1	dog_kennels
select breed_name from breeds group by breed_code order by count(*) desc limit 1	dog_kennels
select t1.owner_id, t1.last_name from owners as t1 join treatments as t2 on t1.owner_id = t2.professional_id group by t1.owner_id order by sum(t2.cost_of_treatment) desc limit 1	dog_kennels
select t1.owner_id, t1.last_name from owners as t1 join treatments as t2 on t1.owner_id = t2.professional_id group by t1.owner_id order by sum(t2.cost_of_treatment) desc limit 1	dog_kennels
select t1.treatment_type_description from treatment_types as t1 join treatments as t2 on t1.treatment_type_code = t2.treatment_type_code group by t2.treatment_type_code order by sum(t2.cost_of_treatment) asc limit 1	dog_kennels
select t2.treatment_type_description from treatments as t1 join treatment_types as t2 on t1.treatment_type_code = t2.treatment_type_code group by t1.treatment_type_code order by sum(t1.cost_of_treatment) limit 1	dog_kennels
select t1.owner_id, t1.zip_code from owners as t1 join dogs as t2 on t1.owner_id = t2.owner_id group by t1.owner_id order by sum(t2.charge_amount	dog_kennels
select t1.owner_id, t1.zip_code from owners as t1 join dogs as t2 on t1.owner_id = t2.owner_id group by t1.owner_id order by sum(t2.cost_of_treatment	dog_kennels
select t1.professional_id, t1.cell_number from professionals as t1 join treatments as t2 on t1.professional_id = t2.professional_id group by t1.professional_id having count(*) >= 2	dog_kennels
select t1.professional_id, t1.cell_number from professionals as t1 join treatments as t2 on t1.professional_id = t2.professional_id group by t1.professional_id having count(*) >= 2	dog_kennels
select t2.first_name, t2.last_name from treatments as t1 join professionals as t2 on t1.professional_id = t2.professional_id where t1.cost_of_treatment < (select avg(cost_of_treatment) from treatments)	dog_kennels
select t2.first_name, t2.last_name from treatments as t1 join professionals as t2 on t1.professional_id = t2.professional_id where t1.cost_of_treatment < (select avg(cost_of_treatment) from treatments)	dog_kennels
select t1.date_of_treatment, t2.first_name from treatments as t1 join professionals as t2 on t1.professional_id = t2.professional_id	dog_kennels
select t1.date_of_treatment, t2.first_name from treatments as t1 join professionals as t2 on t1.professional_id = t2.professional_id	dog_kennels
select t1.cost_of_treatment, t2.treatment_type_description from treatments as t1 join treatment_types as t2 on t1.treatment_type_code = t2.treatment_type_code	dog_kennels
select t1.cost_of_treatment, t2.treatment_type_description from treatments as t1 join treatment_types as t2 on t1.treatment_type_code = t2.treatment_type_code	dog_kennels
select t1.first_name, t1.last_name, t2.size_code from owners as t1 join dogs as t2 on t1.owner_id = t2.owner_id	dog_kennels
select t1.first_name, t1.last_name, t2.size_code from owners as t1 join dogs as t2 on t1.owner_id = t2.owner_id	dog_kennels
select t1.first_name, t2.name from owners as t1 join dogs as t2 on t1.owner_id = t2.dog_id	dog_kennels
select t1.first_name, t2.name from owners as t1 join dogs as t2 on t1.owner_id = t2.owner_id	dog_kennels
select t2.name, t1.date_of_treatment from treatments as t1 join dogs as t2 on t1.dog_id = t2.dog_id where t2.breed_code = (select breed_code from breeds order by count(*) desc limit 1)	dog_kennels
select t2.name, t1.date_of_treatment from treatments as t1 join dogs as t2 on t1.dog_id = t2.dog_id where t2.breed_code = (select breed_code from breeds order by count(*) desc limit 1)	dog_kennels
select t1.first_name, t2.name from owners as t1 join dogs as t2 on t1.owner_id = t2.owner_id where t1.state = "Virginia"	dog_kennels
select t1.first_name, t2.name from owners as t1 join dogs as t2 on t1.owner_id = t2.owner_id where t1.state = "Virginia"	dog_kennels
select t2.date_arrived, t2.date_departed from treatments as t1 join dogs as t2 on t1.dog_id = t2.dog_id	dog_kennels
select t2.date_arrived, t2.date_departed from treatments as t1 join dogs as t2 on t1.dog_id = t2.dog_id	dog_kennels
select t1.last_name from owners as t1 join dogs as t2 on t1.owner_id = t2.owner_id order by t2.age asc limit 1	dog_kennels
select t1.last_name from owners as t1 join dogs as t2 on t1.owner_id = t2.owner_id order by t2.age asc limit 1	dog_kennels
select email_address from professionals where state = "Hawaii" or state = "Wisconsin"	dog_kennels
select email_address from professionals where state = "Hawaii" or state = "Wisconsin"	dog_kennels
select date_arrived, date_departed from dogs	dog_kennels
select date_arrived, date_departed from dogs	dog_kennels
select count(*) from treatments	dog_kennels
select count(*) from treatments	dog_kennels
select count(distinct professional_id) from treatments	dog_kennels
select count(distinct professional_id) from treatments	dog_kennels
select role_code, street, city, state from professionals where city like '%west%'	dog_kennels
select role_code, street, city, state from professionals where city like '%west%'	dog_kennels
select first_name, last_name, email_address from owners where state like '%north%'	dog_kennels
select first_name, last_name, email_address from owners where state like '%north%'	dog_kennels
select count(*) from dogs where age < (select avg(age) from dogs)	dog_kennels
select count(*) from dogs where age < (select avg(age) from dogs)	dog_kennels
select cost_of_treatment from treatments order by date_of_treatment desc limit 1	dog_kennels
select cost_of_treatment from treatments order by date_of_treatment desc limit 1	dog_kennels
select count(*) from dogs where dog_id not in ( select dog_id from treatments )	dog_kennels
select count(*) from dogs where dog_id not in ( select dog_id from treatments )	dog_kennels
select count(*) from owners where owner_id not in ( select owner_id from dogs )	dog_kennels
select count(*) from owners where owner_id not in (select owner_id from dogs)	dog_kennels
select count(*) from professionals where professional_id not in ( select professional_id from treatments )	dog_kennels
select count(*) from professionals where professional_id not in ( select professional_id from treatments )	dog_kennels
select name, age, weight from dogs where abandoned_yn = 1	dog_kennels
select name, age, weight from dogs where abandoned_yn = 1	dog_kennels
select avg(age) from dogs	dog_kennels
select avg(age) from dogs	dog_kennels
select max(age) from dogs	dog_kennels
select age from dogs order by age desc limit 1	dog_kennels
select charge_type, charge_amount from charges group by charge_type	dog_kennels
select charge_type, charge_amount from charges group by charge_type	dog_kennels
select charge_type, sum(charge_amount) from charges order by charge_type desc limit 1	dog_kennels
select charge_amount from charges order by charge_type desc limit 1	dog_kennels
select email_address, cell_number, home_phone from professionals	dog_kennels
select email_address, cell_number, home_phone from professionals	dog_kennels
select breed_code, size_code from dogs	dog_kennels
select distinct breed_code, size_code from dogs	dog_kennels
select t2.first_name, t3.treatment_type_description from treatments as t1 join professionals as t2 on t1.professional_id = t2.professional_id join treatment_types as t3 on t1.treatment_type_code = t3.treatment_type_code	dog_kennels
select t2.first_name, t3.treatment_type_description from treatments as t1 join professionals as t2 on t1.professional_id = t2.professional_id join treatment_types as t3 on t1.treatment_type_code = t3.treatment_type_code	dog_kennels
select count(*) from singer	singer
select count(*) from singer	singer
select name from singer order by net_worth_millions asc	singer
select name from singer order by net_worth_millions asc	singer
select birth_year, citizenship from singer	singer
select birth_year, citizenship from singer	singer
select name from singer where citizenship!= "France"	singer
select name from singer where citizenship!= "France"	singer
select name from singer where birth_year = 1948 or birth_year = 1949	singer
select name from singer where birth_year = 1948 or birth_year = 1949	singer
select name from singer order by net_worth_millions desc limit 1	singer
select name from singer order by net_worth_millions desc limit 1	singer
select citizenship, count(*) from singer group by citizenship	singer
select citizenship, count(*) from singer group by citizenship	singer
select citizenship from singer group by citizenship order by count(*) desc limit 1	singer
select citizenship from singer group by citizenship order by count(*) desc limit 1	singer
select citizenship, max(net_worth_millions) from singer group by citizenship	singer
select max(net_worth_millions), citizenship from singer group by citizenship	singer
select t2.title, t1.name from singer as t1 join song as t2 on t1.singer_id = t2.singer_id	singer
select t2.title, t1.name from singer as t1 join song as t2 on t1.singer_id = t2.singer_id	singer
select distinct t1.name from singer as t1 join song as t2 on t1.singer_id = t2.singer_id where t2.sales > 300000	singer
select distinct t1.name from singer as t1 join song as t2 on t1.singer_id = t2.singer_id where t2.sales > 300000	singer
select t2.name from song as t1 join singer as t2 on t1.singer_id = t2.singer_id group by t1.singer_id having count(*) > 1	singer
select t2.name from song as t1 join singer as t2 on t1.singer_id = t2.singer_id group by t1.singer_id having count(*) > 1	singer
select t2.name, sum(t1.sales) from song as t1 join singer as t2 on t1.singer_id = t2.singer_id group by t1.singer_id	singer
select t1.name, sum(t2.sales) from singer as t1 join song as t2 on t1.singer_id = t2.singer_id group by t1.name	singer
select name from singer where singer_id not in (select singer_id from song)	singer
select name from singer where singer_id not in (select singer_id from song)	singer
select citizenship from singer where birth_year < 1945 intersect select citizenship from singer where birth_year > 1955	singer
select citizenship from singer where birth_year < 1945 intersect select citizenship from singer where birth_year > 1955	singer
select count(*) from other_available_features	real_estate_properties
select t1.feature_type_name from ref_feature_types as t1 join other_available_features as t2 on t1.feature_type_code = t2.feature_type_code where t2.feature_name = "AirCon"	real_estate_properties
select t1.property_type_description from ref_property_types as t1 join properties as t2 on t1.property_type_code = t2.property_type_code	real_estate_properties
select property_name from properties where property_type_code = "House" or property_type_code = "Apartment" and room_count > 1	real_estate_properties
