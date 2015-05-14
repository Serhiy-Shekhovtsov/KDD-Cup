library(ade4)
library(Matrix)
library(plyr)

file_path <- parent.frame(2)$ofile
set.dir <- function () {
	this.dir <- dirname(file_path)
	setwd(this.dir)
}

processing_dir <- "../Data/processing/"
clean_dir <- "../Data/clean/"

data_types <- c("test", "train")
chanks <- c(1, 2, 3, 4, 5)

for_each_chunk <- function(f) {
	for(data_type in data_types) {
		print(paste(" processing", data_type, "data"))
		for (chunk_i in chanks) {
			print(paste("  chunk", chunk_i))
			f(data_type, chunk_i, log=function(...) {
				print(paste0("   ", ...))
			})
		}
	}
}

df.apply <- function (data, FUN) {
	data.frame(unlist(t(sapply(data, FUN=FUN))))
}

# already saved as RDS
save_to_RDS <- function () {
	col_names <- NULL
	# convert data to RDS
	for_each_chunk(function (data_type, chunk_i, log) {
		orig_file_name <- paste0("../Data/src/orange_large_", data_type, ".data.chunk", 
														chunk_i)
		read_header <- ifelse(chunk_i==1, T, F)
		data <- read.table(orig_file_name, sep="\t", header=read_header)
		
		if (is.null(col_names)) {
			col_names <<- colnames(data)
		} else {
			colnames(data) <- col_names
		}
		
		saveRDS(data, paste0(processing_dir, data_type, ".p", chunk_i, ".rds"))
	})
}

save_transformation_info <- function() {
	# gathering info like factors levels, min val, max val, NA rate
	print("gathering info on data")
	
	all_levels <- list()
	col_names <- NULL
	min_vals <- NULL
	max_vals <- NULL
	avg_vals <- NULL
	na_rates <- NULL
	
	for_each_chunk(function(data_type, chunk_i, log) {
	  data <- readRDS(paste0(processing_dir, data_type, ".p", chunk_i, ".rds"))
	  if (is.null(col_names)) {
	  	col_names <<- colnames(data)
	  }
	  
	  # take levels
		data_levels <- Filter(Negate(is.null), sapply(data, levels))
		factor_names <- unique(c(names(all_levels), names(data_levels)))
		
		lapply(factor_names, FUN=function(name){ 
			all_levels[[name]] <<- unique(c(all_levels[[name]], data_levels[[name]]))
		})
		
		data <- data[, !(col_names %in% factor_names)]
		
		# save min, max, avg and NA rate
		c_mins <- df.apply(data, FUN=min)
		min_vals <<- rbind.fill(min_vals, c_mins)
		
		max_vals <<- rbind.fill(max_vals, df.apply(data, FUN=max))
		avg_vals <<- rbind.fill(avg_vals, df.apply(data, FUN=mean))
		
		c_na_rate <- df.apply(data, function (x) { sum(is.na(x)) / length(x) })
		na_rates <<- rbind.fill(na_rates, c_na_rate)
		
		rm(data)
	})
	
	# columns with containing more then 50% NA will be removed
	na_rates <- sapply(na_rates, mean)
	cols_to_remove <- names(na_rates)[na_rates > .5 | is.na(na_rates)]
	
	# save levels
	levels_names <- names(all_levels)
	all_levels <- all_levels[!(levels_names %in% cols_to_remove)]
	saveRDS(all_levels, paste0(processing_dir, "all_levels.rds"))
	
	# save numerics mins and maxs
	num_cols <- col_names[!(col_names %in% cols_to_remove) & !(col_names %in% levels_names)]
	
	min_vals <- sapply(min_vals[, num_cols], FUN=min)
	max_vals <- sapply(max_vals[, num_cols], FUN=max)
	avg_vals <- sapply(avg_vals[, num_cols], FUN=mean)
	
	saveRDS(min_vals, paste0(processing_dir, "min_vals.rds"))
	saveRDS(max_vals, paste0(processing_dir, "max_vals.rds"))
	saveRDS(avg_vals, paste0(processing_dir, "avg_vals.rds"))
}

clean_data = function() {
	print("cleaning data")
	for_each_chunk(function (data_type, chunk_i, log) {
		all_levels = readRDS(paste0(processing_dir, "all_levels.rds"))
		levels_names = names(all_levels)
		data = readRDS(paste0(processing_dir, data_type, ".p", chunk_i, ".rds"))
		
		# transform factors to binary features
		not_num_data = data[, levels_names]
		rm(data)
		
		lapply(names(all_levels), function (name){
			# if this column contains all NA it is treated as logic
			# so we have to convert it
			not_num_data[, name] <<- as.factor(not_num_data[, name])
			levels(not_num_data[,name]) <<- all_levels[[name]]
		})
		rm(all_levels)
		
		log("extracted not_num_data, size=", object.size(not_num_data))
		
		row_indcs = 1:nrow(not_num_data)
		n_parts = 4
		parts = split(row_indcs, ceiling(row_indcs / (length(row_indcs) / n_parts)))
		full_matrix = NULL
		
		for (part_i in 1:length(parts)) {
			log("--- cleaning part ", part_i, " ---")
			part = parts[[part_i]]
			transformed = acm.disjonctif(not_num_data[part,])
			log("factor columns transformed, size = ", object.size(transformed))
			
			log("normalizing numeric data")
			data <- readRDS(paste0(processing_dir, data_type, ".p", chunk_i, ".rds"))
			
			min_vals <- readRDS(paste0(processing_dir, "min_vals.rds"))
			max_vals <- readRDS(paste0(processing_dir, "max_vals.rds"))
			avg_vals <- readRDS(paste0(processing_dir, "avg_vals.rds"))
			num_cols <- colnames(min_vals)
			
			data <- data[part, num_cols]
			
			lapply(num_cols, function (name) {
				col <- data[, name]
				col[is.na(col)] <- avg_vals[name]
				data[, name] <- (col - min_vals[name]) / max_vals[name] - min_vals[name]
			})
			
			log("joining original and transformed data")
			data = cbind(data, transformed)
			rm(transformed)
			
			log("saving data")
			saveRDS(data, paste0(clean_dir, data_type, ".", chunk_i, "_", part_i, ".rds"))
			rm(data)
		}
		
		log("data cleaning completed")
	})
}

do.a.test = function(){
	all_levels = readRDS(paste0(processing_dir, "all_levels.rds"))
	data = readRDS(paste0(processing_dir, "test", ".p", 1, ".rds"))
	
	# transform factors to binary features
	not_num_data = data[, names(all_levels)]
	rm(data)
	
	lapply(names(all_levels), function (name){
		levels(not_num_data[,name]) <<- all_levels[[name]]
	})
	rm(all_levels)
	
	print(paste("extracted not_num_data, size=", object.size(not_num_data)))
	not_num_data
	#not_num_data_transformed = acm.disjonctif(not_num_data)
	#print(paste("not_num_data transformed, size=", object.size(not_num_data_transformed)))
}
