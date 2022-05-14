CREATE TABLE IF NOT EXISTS `final_scores` 
(
    `id` INT NOT NULL AUTO_INCREMENT,
    `user_id` FLOAT DEFAULT NULL,
    `Experience_score` FLOAT DEFAULT NULL,
    `Engagement_score` FLOAT DEFAULT NULL,
    `Satisfactory_score` FLOAT DEFAULT NULL,
    PRIMARY KEY (`id`)
)
ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE utf8mb4_unicode_ci;
