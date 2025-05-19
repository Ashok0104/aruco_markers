import 'dart:convert';
import 'dart:io';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:image_picker/image_picker.dart';
import 'package:http_parser/http_parser.dart';

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ArUco Detector',
      theme: ThemeData(
        primarySwatch: Colors.blue,
        visualDensity: VisualDensity.adaptivePlatformDensity,
      ),
      home: ArucoDetectorPage(),
    );
  }
}

class ArucoDetectorPage extends StatefulWidget {
  @override
  _ArucoDetectorPageState createState() => _ArucoDetectorPageState();
}

class _ArucoDetectorPageState extends State<ArucoDetectorPage> {
  final ImagePicker _picker = ImagePicker();
  File? _selectedImage;
  bool _isLoading = false;
  bool _isProcessing = false;
  String? _detectionResultId;
  String? _sequentialOutput;
  List<dynamic>? _detectedMarkers;
  List<List<dynamic>>? _rows;
  String? _errorMessage;
  // API configuration
  final String apiBaseUrl =
      "http://192.168.4.73:8000"; // Update with your API URL

  Future<void> _pickImage(ImageSource source) async {
    try {
      final XFile? pickedFile = await _picker.pickImage(source: source);

      if (pickedFile != null) {
        setState(() {
          _selectedImage = File(pickedFile.path);
          // Reset previous results
          _detectionResultId = null;
          _sequentialOutput = null;
          _detectedMarkers = null;
          _rows = null;
          _errorMessage = null;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = "Error picking image: $e";
      });
    }
  }

  Future<void> _uploadImage() async {
    if (_selectedImage == null) {
      setState(() {
        _errorMessage = "Please select an image first";
      });
      return;
    }

    setState(() {
      _isProcessing = true;
      _errorMessage = null;
    });

    try {
      // Create multipart request
      var request = http.MultipartRequest(
        'POST',
        Uri.parse('$apiBaseUrl/detect_markers/'),
      );

      // Determine source type based on how the image was selected
      String sourceType =
          _imageSource == ImageSource.camera ? "camera" : "gallery";

      // Add query parameters
      request.fields['source'] = sourceType;

      // Add dictionary types as query parameters
      request.files.add(
        await http.MultipartFile.fromPath(
          'file',
          _selectedImage!.path,
          contentType: MediaType('image', 'jpeg'),
        ),
      );

      // Add the ArUco dictionary types
      request.fields['dict_types'] = 'DICT_4X4_50,DICT_6X6_250';

      // Send the request
      var streamedResponse = await request.send();
      var response = await http.Response.fromStream(streamedResponse);

      if (response.statusCode == 200) {
        var data = jsonDecode(response.body);
        setState(() {
          _detectionResultId = data['result_id'];
          _sequentialOutput = data['sequential_output'];
          _detectedMarkers = data['markers'];
          _rows = List<List<dynamic>>.from(
              data['rows'].map((row) => List<dynamic>.from(row)));
          _isProcessing = false;
        });
      } else {
        setState(() {
          _errorMessage = "Error: ${response.statusCode} - ${response.body}";
          _isProcessing = false;
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = "Error uploading image: $e";
        _isProcessing = false;
      });
    }
  }

  Future<void> _sendBluetooth() async {
    if (_sequentialOutput == null || _sequentialOutput!.isEmpty) {
      setState(() {
        _errorMessage = "No sequential output to send";
      });
      return;
    }

    setState(() {
      _isLoading = true;
      _errorMessage = null;
    });

    try {
      // Create the request body with the sequential output message directly
      final requestBody = jsonEncode({"message": _sequentialOutput});

      final response = await http.post(
        Uri.parse('$apiBaseUrl/send_bluetooth_direct/'),
        headers: {
          'Content-Type': 'application/json',
        },
        body: requestBody,
      );

      if (response.statusCode == 200) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text("Bluetooth sent: $_sequentialOutput")),
        );
      } else {
        setState(() {
          _errorMessage =
              "Error sending Bluetooth: ${response.statusCode} - ${response.body}";
        });
      }
    } catch (e) {
      setState(() {
        _errorMessage = "Error with Bluetooth: $e";
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  ImageSource? _imageSource;

  void _showImageSourceSelector() {
    showModalBottomSheet(
      context: context,
      builder: (context) => SafeArea(
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            ListTile(
              leading: Icon(Icons.camera_alt),
              title: Text('Take a photo'),
              onTap: () {
                Navigator.of(context).pop();
                setState(() {
                  _imageSource = ImageSource.camera;
                });
                _pickImage(ImageSource.camera);
              },
            ),
            ListTile(
              leading: Icon(Icons.photo_library),
              title: Text('Choose from gallery'),
              onTap: () {
                Navigator.of(context).pop();
                setState(() {
                  _imageSource = ImageSource.gallery;
                });
                _pickImage(ImageSource.gallery);
              },
            ),
          ],
        ),
      ),
    );
  }

  String interpretSequentialOutput(String output) {
    if (output == null || output.isEmpty) {
      return "No commands";
    }

    Map<String, String> interpretations = {
      'F': 'Forward',
      'B': 'Backward',
      'L': 'Left',
      'R': 'Right',
    };

    List<String> result = [];
    int count = 1;
    String currentChar = output[0];

    for (int i = 1; i < output.length; i++) {
      if (output[i] == currentChar) {
        count++;
      } else {
        if (interpretations.containsKey(currentChar)) {
          result.add(
              "${interpretations[currentChar]} ${count > 1 ? 'x $count' : ''}");
        } else {
          result.add("$currentChar ${count > 1 ? 'x $count' : ''}");
        }
        currentChar = output[i];
        count = 1;
      }
    }

    // Add the last group
    if (interpretations.containsKey(currentChar)) {
      result.add(
          "${interpretations[currentChar]} ${count > 1 ? 'x $count' : ''}");
    } else {
      result.add("$currentChar ${count > 1 ? 'x $count' : ''}");
    }

    return result.join('\n');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        centerTitle: true,
        title: Text(
          'A R U C O',
          style: TextStyle(
            fontSize: 30,
            fontWeight: FontWeight.bold,
            color: Colors.black,
          ),
        ),
        shape: RoundedRectangleBorder(
          borderRadius: BorderRadius.vertical(
            bottom: Radius.circular(16),
          ),
        ),
      ),
      backgroundColor: Colors.amber.withOpacity(0.9),
      body: SingleChildScrollView(
        child: Padding(
          padding: const EdgeInsets.all(16.0),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              // Image selection area
              Container(
                height: 300,
                decoration: BoxDecoration(
                  color: Colors.transparent,
                  borderRadius: BorderRadius.circular(8.0),
                ),
                child: _selectedImage != null
                    ? Image.file(_selectedImage!, fit: BoxFit.contain)
                    : Center(child: Text('No image selected')),
              ),
              SizedBox(height: 16),

              // Image selection button
              ElevatedButton.icon(
                onPressed: _showImageSourceSelector,
                icon: Icon(
                  Icons.add_photo_alternate,
                  color: Colors.black,
                ),
                label: Text(
                  'Select Image',
                  style: TextStyle(color: Colors.black),
                ),
                style: ElevatedButton.styleFrom(
                  padding: EdgeInsets.symmetric(vertical: 12.0),
                ),
              ),
              SizedBox(height: 8),

              // Upload button
              if (_selectedImage != null)
                ElevatedButton(
                  onPressed: _isProcessing ? null : _uploadImage,
                  child: _isProcessing
                      ? Row(
                          mainAxisAlignment: MainAxisAlignment.center,
                          children: [
                            SizedBox(
                              height: 16,
                              width: 16,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            ),
                            SizedBox(width: 8),
                            Text('Processing...'),
                          ],
                        )
                      : Text('Submit',
                          style: TextStyle(fontSize: 18, color: Colors.black)),
                  style: ElevatedButton.styleFrom(
                    padding: EdgeInsets.symmetric(vertical: 12.0),
                    backgroundColor: Colors.green,
                  ),
                ),
              SizedBox(height: 16),

              // Error message
              if (_errorMessage != null)
                Container(
                  padding: EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.red[100],
                    borderRadius: BorderRadius.circular(4),
                  ),
                  child: Text(
                    _errorMessage!,
                    style: TextStyle(color: Colors.red[900]),
                  ),
                ),

              // Results section
              if (_sequentialOutput != null) ...[
                Divider(thickness: 1),
                Text(
                  'Detection Results',
                  style: TextStyle(
                    fontSize: 18,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                SizedBox(height: 8),

                // Sequential output with Bluetooth option
                // Card(
                //   elevation: 2,
                //   child: Padding(
                //     padding: const EdgeInsets.all(16.0),
                //     child: Row(
                //       children: [
                //         Expanded(
                //           child: Column(
                //             crossAxisAlignment: CrossAxisAlignment.start,
                //             children: [
                //               Text(
                //                 'Sequential Output:',
                //                 style: TextStyle(
                //                   fontWeight: FontWeight.bold,
                //                   color: Colors.grey[600],
                //                 ),
                //               ),
                //               SizedBox(height: 4),
                //               Text(
                //                 _sequentialOutput!,
                //                 style: TextStyle(
                //                   fontSize: 24,
                //                   fontWeight: FontWeight.bold,
                //                 ),
                //               ),
                //             ],
                //           ),
                //         ),
                //         IconButton(
                //           icon: Icon(Icons.bluetooth, color: Colors.blue),
                //           onPressed: _isLoading ? null : _sendBluetooth,
                //           tooltip: 'Send via Bluetooth',
                //         ),
                //         if (_isLoading)
                //           SizedBox(
                //             height: 16,
                //             width: 16,
                //             child: CircularProgressIndicator(strokeWidth: 2),
                //           ),
                //       ],
                //     ),
                //   ),
                // ),
                SizedBox(height: 16),
                Card(
                  elevation: 2,
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Row(
                          children: [
                            Expanded(
                              child: Column(
                                crossAxisAlignment: CrossAxisAlignment.start,
                                children: [
                                  Text(
                                    'Moves:',
                                    style: TextStyle(
                                      fontSize: 22,
                                      fontWeight: FontWeight.bold,
                                      color: Colors.grey[600],
                                    ),
                                  ),
                                ],
                              ),
                            ),
                            IconButton(
                              icon: Icon(Icons.bluetooth, color: Colors.blue),
                              onPressed: _isLoading ? null : _sendBluetooth,
                              tooltip: 'Send via Bluetooth',
                            ),
                            if (_isLoading)
                              SizedBox(
                                height: 16,
                                width: 16,
                                child:
                                    CircularProgressIndicator(strokeWidth: 2),
                              ),
                          ],
                        ),
                        SizedBox(height: 12),
                        Center(
                          child: Text(
                            interpretSequentialOutput(_sequentialOutput!),
                            textAlign: TextAlign.center,
                            style: TextStyle(
                              fontSize: 20,
                              color: Colors.green[800],
                            ),
                          ),
                        ),
                      ],
                    ),
                  ),
                ),

                // Detected markers
                // if (_detectedMarkers != null &&
                //     _detectedMarkers!.isNotEmpty) ...[
                //   Text(
                //     'Detected Markers:',
                //     style: TextStyle(
                //       fontWeight: FontWeight.bold,
                //     ),
                //   ),
                //   SizedBox(height: 8),
                //   // Display marker grid
                //   if (_rows != null)
                //     Card(
                //       child: Padding(
                //         padding: const EdgeInsets.all(8.0),
                //         child: Column(
                //           children: _rows!.map((row) {
                //             return Padding(
                //               padding:
                //                   const EdgeInsets.symmetric(vertical: 4.0),
                //               child: Row(
                //                 mainAxisAlignment:
                //                     MainAxisAlignment.spaceEvenly,
                //                 children: row.map((marker) {
                //                   return Container(
                //                     width: 40,
                //                     height: 40,
                //                     decoration: BoxDecoration(
                //                       color: Colors.blue[100],
                //                       borderRadius: BorderRadius.circular(4),
                //                     ),
                //                     child: Center(
                //                       child: Text(
                //                         marker.toString(),
                //                         style: TextStyle(
                //                           fontWeight: FontWeight.bold,
                //                           fontSize: 18,
                //                         ),
                //                       ),
                //                     ),
                //                   );
                //                 }).toList(),
                //               ),
                //             );
                //           }).toList(),
                //         ),
                //       ),
                //     ),
                // ],
              ],
            ],
          ),
        ),
      ),
    );
  }
}
