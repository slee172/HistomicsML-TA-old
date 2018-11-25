<?php

//
//	Copyright (c) 2014-2015, Emory University
//	All rights reserved.
//
//	Redistribution and use in source and binary forms, with or without modification, are
//	permitted provided that the following conditions are met:
//
//	1. Redistributions of source code must retain the above copyright notice, this list of
//	conditions and the following disclaimer.
//
//	2. Redistributions in binary form must reproduce the above copyright notice, this list 
// 	of conditions and the following disclaimer in the documentation and/or other materials
//	provided with the distribution.
//
//	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
//	EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
//	OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
//	SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
//	INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED 
//	TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
//	BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
//	CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
//	WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
//	DAMAGE.
//
//

	require '../db/logging.php';		// Also includes connect.php
	require 'hostspecs.php';

	session_start();

	$dataSet = $_POST['dataset'];
	$trainSet = $_POST['trainset'];

	// Get the dataset file from the database
	//
	$dbConn = guestConnect();
	$sql = 'SELECT features_file FROM datasets WHERE name="'.$dataSet.'"';

	if( $result = mysqli_query($dbConn, $sql) ) {

		$featureFile = mysqli_fetch_row($result);			
		mysqli_free_result($result);
	}

	$sql = 'SELECT filename FROM training_sets WHERE name="'.$trainSet.'"';

	if( $result = mysqli_query($dbConn, $sql) ) {

		$trainSetFile = mysqli_fetch_row($result);			
		mysqli_free_result($result);
	}

	mysqli_close($dbConn);

	if( $_SESSION['uid'] === null ) {
		write_log("INFO", "No session active");

		$UID = uniqid("", true);
		$_SESSION['uid'] = $UID;

	}	


	$cmd =  array( "command" => "viewerLoad", 
				   "uid" => $_SESSION['uid'],
			 	   "dataset" => $featureFile[0],
			 	   "trainset" => $trainSetFile[0]
		 	   );
				 	
   
	$cmd = json_encode($cmd);
		
	$addr = gethostbyname($host);
	set_time_limit(0);
	
	$socket = socket_create(AF_INET, SOCK_STREAM, 0);
	if( $socket === false ) {
		log_error("socket_create failed:  ". socket_strerror(socket_last_error()));
	}
	
	$result = socket_connect($socket, $addr, $port);
	if( !$result ) {
		log_error("socket_connect failed: ".socket_strerror(socket_last_error()));
	}	
	socket_write($socket, $cmd, strlen($cmd));	
	
	$response = socket_read($socket, 8192);
	echo $response;
?>
